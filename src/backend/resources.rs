use ash::vk;
use gpu_allocator::vulkan::Allocation;
use smallvec::SmallVec;

pub(crate) struct InnerBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) mem_requirements: vk::MemoryRequirements,
    pub(crate) device_address: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) allocation: Allocation,
}

pub(crate) struct InnerImageView {
    pub(crate) view: vk::ImageView,
    pub(crate) subresources: vk::ImageSubresourceRange,
}

// need to handle views some how tho
pub(crate) struct InnerImage {
    pub(crate) image: ash::vk::Image,
    pub(crate) mem_requirements: vk::MemoryRequirements,
    pub(crate) allocation: Allocation,
    pub(crate) format: vk::Format,

    // idk how to deal with ts..
    pub(crate) image_views: SmallVec<[InnerImageView; 4]>,
}
