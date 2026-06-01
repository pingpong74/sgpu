use ash::vk;
use gpu_allocator::vulkan::Allocation;

pub(crate) struct InnerBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) mem_requirements: vk::MemoryRequirements,
    pub(crate) device_address: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) allocation: Allocation,
}

// need to handle views some how tho
pub(crate) struct InnerImage {
    pub(crate) image: ash::vk::Image,
    pub(crate) mem_requirements: vk::MemoryRequirements,
    pub(crate) allocation: Allocation,
}
