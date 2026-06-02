use std::{
    mem::ManuallyDrop,
    sync::{Mutex, atomic::AtomicU64},
};

use crate::{
    BufferDescription, MemoryType,
    api::SgpuInititizationInfo,
    backend::{commands::Queue, instance::Instance, physical_device::PhysicalDevice, resources::InnerBuffer},
    commands::QueueType,
};
use ash::vk::{self, SemaphoreWaitInfo};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};

pub(crate) struct Device {
    pub(crate) handle: ash::Device,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) allocator: ManuallyDrop<Mutex<Allocator>>,
    pub(crate) queue_indices: [u32; 3],
}

impl Device {
    pub(crate) fn new(sgpu_init_info: &SgpuInititizationInfo, instance: &Instance) -> Device {
        // Required device extensions (swapchain needed for presentation)
        let mut device_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::synchronization2::NAME.as_ptr(),
        ];

        if sgpu_init_info.ray_tracing {
            device_extensions.push(ash::khr::acceleration_structure::NAME.as_ptr());
            device_extensions.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            device_extensions.push(ash::khr::deferred_host_operations::NAME.as_ptr());
            if !device_extensions.contains(&ash::khr::spirv_1_4::NAME.as_ptr()) {
                device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
            }
        }

        if sgpu_init_info.atomic_float_operations {
            device_extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }

        if sgpu_init_info.mesh_shaders {
            device_extensions.push(ash::ext::mesh_shader::NAME.as_ptr());
            device_extensions.push(ash::khr::shader_float_controls::NAME.as_ptr());
            if !device_extensions.contains(&ash::khr::spirv_1_4::NAME.as_ptr()) {
                device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
            }
        }

        let physical_device = {
            let dev = PhysicalDevice::select_physical_device(&instance, &device_extensions);
            if dev.is_none() {
                panic!("Failed to find vulkan compatible device")
            }

            dev.unwrap()
        };

        let unique_families: Vec<u32> = {
            let mut v: Vec<_> = physical_device.queue_families.queue_families_indices.iter().map(|q| q.unwrap()).collect();
            v.sort();
            v.dedup();
            v
        };

        // Queue priorities (all same)
        let priorities = [1.0_f32];
        let queue_infos: Vec<_> = unique_families
            .iter()
            .map(|&family| vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities))
            .collect();

        // Existing common features
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true).multi_draw_indirect(true).sampler_anisotropy(true);
        let mut float_atomic_features = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default().shader_buffer_float32_atomic_add(true);

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_storage_texel_buffer_update_after_bind(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true)
            .descriptor_binding_uniform_texel_buffer_update_after_bind(true);

        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut timeline_sem = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut vk_features_11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true)
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true);

        // Ray tracing
        let mut accel_struct_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();

        if sgpu_init_info.ray_tracing {
            accel_struct_features = accel_struct_features.acceleration_structure(true);
            rt_pipeline_features = rt_pipeline_features.ray_tracing_pipeline(true);
            ray_query_features = ray_query_features.ray_query(true);
        }

        // mesh shaders
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();

        if sgpu_init_info.mesh_shaders {
            mesh_shader_features = mesh_shader_features.mesh_shader(true).task_shader(true);
        }

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut indexing_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut sync2)
            .push_next(&mut timeline_sem)
            .push_next(&mut buffer_device_address)
            .push_next(&mut vk_features_11)
            .push_next(&mut mesh_shader_features)
            .features(features);

        if sgpu_init_info.ray_tracing {
            features2 = features2.push_next(&mut accel_struct_features).push_next(&mut rt_pipeline_features).push_next(&mut ray_query_features);
        }

        if sgpu_init_info.atomic_float_operations {
            features2 = features2.push_next(&mut float_atomic_features);
        }

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        let dev = unsafe { instance.handle.create_device(physical_device.handle, &create_info, None).expect("Failed to create logical device") };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle.clone(),
            device: dev.clone(),
            physical_device: physical_device.handle,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .expect("Failed to create allocator");

        return Device {
            queue_indices: std::array::from_fn(|i| physical_device.queue_families.queue_families_indices[i].unwrap()),
            handle: dev,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            physical_device: physical_device,
        };
    }

    pub(crate) fn get_queues(&self) -> [Queue; 3] {
        unsafe {
            let create_semaphore = {
                let mut type_info = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);

                let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

                self.handle.create_semaphore(&create_info, None).expect("Failed to create timeline semaphore")
            };

            return std::array::from_fn(|i| Queue {
                queue: self.handle.get_device_queue(self.physical_device.queue_families.queue_families_indices[i].unwrap(), 0),
                queue_type: QueueType::Graphics,
                semaphore: create_semaphore,
                cpu_signaled_value: Mutex::new(0),
            });
        }
    }

    pub(crate) fn create_buffer(&self, desc: &BufferDescription) -> InnerBuffer {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(desc.usage.to_vk() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .size(desc.size)
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&self.queue_indices);

        let buffer = unsafe { self.handle.create_buffer(&buffer_create_info, None).expect("Failed to create buffer ") };
        let memory_requirements = unsafe { self.handle.get_buffer_memory_requirements(buffer) };

        let allocation_create_info = AllocationCreateDesc {
            name: "o",
            requirements: memory_requirements,
            location: desc.memory_type.to_vk_flag(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = { self.allocator.lock().unwrap().allocate(&allocation_create_info).expect("Failed to allocate memory on device") };

        unsafe {
            self.handle.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).expect("Failed to bind buffer memory");
        }

        let buffer_address = unsafe { self.handle.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };

        let mapper_ptr = match desc.memory_type {
            MemoryType::HostVisible | MemoryType::PreferHost => Some(allocation.mapped_ptr().unwrap().as_ptr() as *mut u8),
            _ => None,
        };

        return InnerBuffer {
            buffer: buffer,
            mem_requirements: memory_requirements,
            device_address: buffer_address,
            mapped_ptr: mapper_ptr,
            allocation,
        };
    }

    pub(crate) fn destroy_buffer(&self, buffer: InnerBuffer) {
        unsafe {
            self.handle.destroy_buffer(buffer.buffer, None);
            self.allocator.lock().unwrap().free(buffer.allocation).expect("Failed to free buffer");
        }
    }

    #[inline]
    pub(crate) fn create_command_pool(&self, queue_type: QueueType) -> vk::CommandPool {
        return unsafe {
            self.handle
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_indices[queue_type as usize]), None)
                .expect("Failed to create command pool")
        };
    }

    #[inline]
    pub(crate) fn destroy_command_pool(&self, pool: vk::CommandPool) {
        unsafe {
            if pool != vk::CommandPool::null() {
                self.handle.destroy_command_pool(pool, None);
            }
        }
    }

    pub(crate) fn wait_queue(&self, queue: &Queue, value: u64) {
        unsafe {
            self.handle
                .wait_semaphores(&SemaphoreWaitInfo::default().semaphores(&[queue.semaphore]).values(&[value]), u64::MAX)
                .expect("Failed to wait on Semaphore");
        }
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
            self.handle.destroy_device(None);
        }
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}
