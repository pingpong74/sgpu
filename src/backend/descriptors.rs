use super::device::Device;
use ash::vk;

pub const SAMPLED_IMAGE_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;
pub const BINDLESS_BUFFER_BINDING: u32 = 2;

pub const BINDLESS_MAX_SAMPLERS: u32 = 16384;
pub const BINDLESS_MAX_STORAGE: u32 = 16384;
pub const BINDLESS_MAX_BUFFERS: u32 = 65536;

pub(crate) struct BindlessDescriptorSet {
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl BindlessDescriptorSet {
    pub(crate) fn new(device: &Device) -> BindlessDescriptorSet {
        let flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(SAMPLED_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(BINDLESS_MAX_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(STORAGE_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(BINDLESS_MAX_STORAGE)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDLESS_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(BINDLESS_MAX_BUFFERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let layout = unsafe {
            device
                .handle
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&bindings)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(&mut binding_flags),
                    None,
                )
                .expect("bindless layout creation failed")
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: BINDLESS_MAX_SAMPLERS,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: BINDLESS_MAX_STORAGE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: BINDLESS_MAX_BUFFERS,
            },
        ];

        let pool = unsafe {
            device
                .handle
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&pool_sizes)
                        .max_sets(1)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .expect("bindless pool creation failed")
        };

        let set = unsafe {
            device
                .handle
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&layout)))
                .expect("bindless set allocation failed")[0]
        };

        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: 256,
        };

        let pipeline_layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[layout]).push_constant_ranges(&[push_range]), None)
                .unwrap()
        };

        return BindlessDescriptorSet {
            layout,
            pool,
            set,
            pipeline_layout,
        };
    }

    pub(crate) fn write_buffer(&self, device: &Device, buffer: vk::Buffer, index: u32) {
        let buffer_info = [
            vk::DescriptorBufferInfo {
                buffer: buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];

        let write_info = [
            vk::WriteDescriptorSet::default()
                .buffer_info(&buffer_info)
                .dst_set(self.set)
                .dst_binding(BINDLESS_BUFFER_BINDING)
                .dst_array_element(index)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER),
        ];

        unsafe {
            device.handle.update_descriptor_sets(&write_info, &[]);
        }
    }

    pub(crate) fn write_sampled_image(&self, device: &Device, image_view: vk::ImageView, index: u32) {
        let sampler_info = [
            vk::DescriptorImageInfo {
                image_view: image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler: vk::Sampler::null(),
            },
        ];

        let write_info = [
            vk::WriteDescriptorSet::default()
                .image_info(&sampler_info)
                .dst_set(self.set)
                .dst_binding(SAMPLED_IMAGE_BINDING)
                .dst_array_element(index)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE),
        ];

        unsafe {
            device.handle.update_descriptor_sets(&write_info, &[]);
        }
    }

    pub(crate) fn write_storage_image(&self, device: &Device, image_view: vk::ImageView, index: u32) {
        let sampler_info = [
            vk::DescriptorImageInfo {
                image_view: image_view,
                image_layout: vk::ImageLayout::GENERAL,
                sampler: vk::Sampler::null(),
            },
        ];

        let write_info = [
            vk::WriteDescriptorSet::default()
                .image_info(&sampler_info)
                .dst_set(self.set)
                .dst_binding(STORAGE_IMAGE_BINDING)
                .dst_array_element(index)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE),
        ];

        unsafe {
            device.handle.update_descriptor_sets(&write_info, &[]);
        }
    }

    pub(crate) fn cleanup(&self, device: &Device) {
        unsafe {
            device.handle.destroy_descriptor_set_layout(self.layout, None);
            device.handle.destroy_descriptor_pool(self.pool, None);
            device.handle.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
