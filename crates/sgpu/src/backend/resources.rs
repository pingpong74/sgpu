use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::ImageUsage;

use std::sync::Arc;

pub(crate) enum AllocationSource {
    Dedicated(Allocation),
    Aliased(Arc<Allocation>),
}

pub(crate) struct InnerBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) allocation: AllocationSource,
}

pub(crate) struct InnerImageView {
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,
    pub(crate) subresources: vk::ImageSubresourceRange,
}

pub(crate) struct InnerImage {
    pub(crate) image: ash::vk::Image,
    pub(crate) allocation: AllocationSource,
    pub(crate) format: vk::Format,
    pub(crate) usage: ImageUsage,
}
