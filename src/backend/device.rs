use std::{mem::ManuallyDrop, sync::Mutex};

use crate::{
    BufferDescription, ComputePipeline, ImageDescription, ImageViewDescription, MemoryType,
    api::SgpuInititizationInfo,
    backend::{
        commands::Queue,
        descriptors::BindlessDescriptorSet,
        instance::Instance,
        physical_device::PhysicalDevice,
        resources::{InnerBuffer, InnerImage, InnerImageView},
    },
    commands::QueueType,
    pipeline::{RasterizationPipeline, RasterizationPipelineDescription},
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
        }

        if sgpu_init_info.atomic_float_operations {
            device_extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }

        if sgpu_init_info.mesh_shaders {
            device_extensions.push(ash::ext::mesh_shader::NAME.as_ptr());
            device_extensions.push(ash::khr::shader_float_controls::NAME.as_ptr());
        }

        if sgpu_init_info.mesh_shaders | sgpu_init_info.ray_tracing {
            device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
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

        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut vk_features_11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true)
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true);

        let mut vk_features_12 = vk::PhysicalDeviceVulkan12Features::default()
            .draw_indirect_count(true)
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .descriptor_indexing(true)
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
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut sync2)
            .push_next(&mut vk_features_11)
            .push_next(&mut vk_features_12)
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
            let create_semaphore = || {
                let mut type_info = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);

                let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

                self.handle.create_semaphore(&create_info, None).expect("Failed to create timeline semaphore")
            };

            return std::array::from_fn(|i| Queue {
                queue: self.handle.get_device_queue(self.physical_device.queue_families.queue_families_indices[i].unwrap(), 0),
                queue_type: QueueType::QUEUE_TYPES[i],
                semaphore: create_semaphore(),
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
            location: desc.memory_type.to_vk(),
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
            size: desc.size,
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

    pub(crate) fn create_image(&self, desc: &ImageDescription) -> InnerImage {
        let image_create_info = vk::ImageCreateInfo::default()
            .flags(desc.create_flags.to_vk())
            .usage(desc.usage.to_vk())
            .extent(desc.extent.to_vk())
            .format(desc.format.to_vk())
            .array_layers(desc.array_layers)
            .mip_levels(desc.mip_levels)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .image_type(desc.image_type.to_vk())
            .samples(desc.samples.to_vk())
            .tiling(vk::ImageTiling::OPTIMAL);

        let image = unsafe { self.handle.create_image(&image_create_info, None).expect("Failed to create Image") };

        let memory_requirements = unsafe { self.handle.get_image_memory_requirements(image) };

        let allocation_create_info = AllocationCreateDesc {
            name: "o",
            requirements: memory_requirements,
            location: desc.memory_type.to_vk(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = self.allocator.lock().unwrap().allocate(&allocation_create_info).expect("Failed to allocate memory on device");

        unsafe {
            self.handle.bind_image_memory(image, allocation.memory(), allocation.offset()).expect("Failed to bind image memory");
        }

        return InnerImage {
            image: image,
            mem_requirements: memory_requirements,
            allocation: allocation,
            format: desc.format.to_vk(),
        };
    }

    pub(crate) fn create_image_view(&self, image: &InnerImage, desc: &ImageViewDescription) -> InnerImageView {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image.image)
            .view_type(desc.view_type.to_vk())
            .format(image.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(desc.subresources.to_vk_subresource_range());

        let view = unsafe { self.handle.create_image_view(&image_view_create_info, None).expect("Failed to create Image view") };

        return InnerImageView {
            view: view,
            image: image.image,
            subresources: desc.subresources.to_vk_subresource_range(),
        };
    }

    pub(crate) fn destroy_image_view(&self, image_view: InnerImageView) {
        unsafe {
            self.handle.destroy_image_view(image_view.view, None);
        }
    }

    pub(crate) fn destroy_image(&self, image: InnerImage) {
        unsafe {
            self.handle.destroy_image(image.image, None);
            self.allocator.lock().unwrap().free(image.allocation).expect("Failed to free buffer");
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

    pub(crate) fn create_raster_pipeline(&self, layout: &BindlessDescriptorSet, desc: &RasterizationPipelineDescription) -> RasterizationPipeline {
        assert!(!desc.vertex_shader.is_empty(), "vertex shader SPIR-V is empty");
        assert!(!desc.fragment_shader.is_empty(), "fragment shader SPIR-V is empty");

        let entry = std::ffi::CString::new("main").unwrap();

        let vert = self.create_shader_module(desc.vertex_shader);
        let frag = self.create_shader_module(desc.fragment_shader);

        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&entry),
        ];

        // Bindless — no vertex bindings or attributes
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology.to_vk()).primitive_restart_enable(false);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(desc.polygon_mode.to_vk())
            .cull_mode(desc.cull_mode.to_vk())
            .front_face(desc.front_face.to_vk())
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_stencil.depth_test)
            .depth_write_enable(desc.depth_stencil.depth_write)
            .depth_compare_op(desc.depth_stencil.depth_compare.to_vk())
            .depth_bounds_test_enable(false)
            .stencil_test_enable(desc.depth_stencil.stencil_test);

        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc.outputs.color.iter().map(|_| desc.blend_mode.to_vk_attachment()).collect();

        let blend = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(&blend_attachments);

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);

        let color_formats: Vec<vk::Format> = desc.outputs.color.iter().map(|f| f.to_vk()).collect();

        let depth_format = desc.outputs.depth.map(|f| f.to_vk()).unwrap_or(vk::Format::UNDEFINED);

        let stencil_format = desc.outputs.stencil.map(|f| f.to_vk()).unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout.pipeline_layout)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            self.handle
                .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&create_info), None)
                .expect("raster pipeline creation failed")[0]
        };

        unsafe {
            self.handle.destroy_shader_module(vert, None);
            self.handle.destroy_shader_module(frag, None);
        }

        return RasterizationPipeline { handle };
    }

    fn create_shader_module(&self, spirv: &[u8]) -> vk::ShaderModule {
        assert!(spirv.len() % 4 == 0, "SPIR-V must be 4-byte aligned");

        let words: Vec<u32> = spirv.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        unsafe {
            self.handle
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .expect("shader module creation failed")
        }
    }

    #[inline]
    pub(crate) fn wait_idle(&self) {
        unsafe {
            self.handle.device_wait_idle().unwrap();
        }
    }

    pub(crate) fn create_compute_pipeline(&self, layout: &BindlessDescriptorSet, shader: &[u8]) -> ComputePipeline {
        let entry = std::ffi::CString::new("main").unwrap();
        let shader = self.create_shader_module(shader);

        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(&entry);

        let pipeline_info = [vk::ComputePipelineCreateInfo::default().layout(layout.pipeline_layout).stage(shader_stage_info)];

        let pipeline = unsafe {
            self.handle
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .expect("Failed to create compute pipeline")
        }[0];

        unsafe {
            self.handle.destroy_shader_module(shader, None);
        }

        return ComputePipeline {
            handle: pipeline,
        };
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
