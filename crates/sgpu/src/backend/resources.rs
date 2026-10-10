use ash::vk;
use gpu_allocator::vulkan::Allocation;

#[allow(unused)]
pub(crate) struct InnerBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: u64,
    pub(crate) device_address: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) allocation: Allocation,
}

pub(crate) struct InnerImageView {
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,
    pub(crate) subresources: vk::ImageSubresourceRange,
}

// need to handle views some how tho
#[allow(unused)]
pub(crate) struct InnerImage {
    pub(crate) image: ash::vk::Image,
    pub(crate) mem_requirements: vk::MemoryRequirements,
    pub(crate) allocation: Allocation,
    pub(crate) format: vk::Format,
}
