use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, ImageView, QueueType, SampledImageId, StorageImageId};

use crate::access::{BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::resource::{Id, PassBuffer, PassImage};

/// This represents a bit of GPU work
pub trait Node<R, D> {
    /// Queue this work will run on
    const QUEUE: QueueType = QueueType::Graphics;
    /// Additional Information used to initialize the node
    type Config;

    /// All the resources to be used are qccquired here using `NodeWritter`
    fn declare(resources: &R, config: &Self::Config, write: &mut NodeWriter) -> Self;

    /// Actual command to be executed onthe GPU
    fn execute(&self, data: &D, context: &mut ExecutionContext<'_>);
}

/// Used to register the access's of resources
pub struct NodeWriter {
    pub(crate) image_count: usize,
    pub(crate) buffer_count: usize,
    pub(crate) images: Vec<(u32, AccessType)>,
    pub(crate) buffers: Vec<(u32, AccessType)>,
}

impl NodeWriter {
    pub(crate) fn new(image_count: usize, buffer_count: usize) -> Self {
        NodeWriter {
            image_count,
            buffer_count,
            images: Vec::new(),
            buffers: Vec::new(),
        }
    }

    /// Get Image with given Access
    pub fn image<A: ImageAccess>(&mut self, id: Id<Image>) -> PassImage<A> {
        assert!((id.index as usize) < self.image_count, "image id {id:?} does not belong to this graph");
        self.images.push((id.index, A::ACCESS));
        PassImage::new(id.index)
    }

    /// Get Buffer with given Access
    pub fn buffer<A: BufferAccess>(&mut self, id: Id<Buffer>) -> PassBuffer<A> {
        assert!((id.index as usize) < self.buffer_count, "buffer id {id:?} does not belong to this graph");
        self.buffers.push((id.index, A::ACCESS));
        PassBuffer::new(id.index)
    }
}

/// Consists of a CommandBuffer
/// Used to get `sgpu::Image` and `sgpu::Buffer`
pub struct ExecutionContext<'a> {
    pub(crate) cmd: &'a mut CommandBuffer,
    pub(crate) images: &'a [Image],
    pub(crate) buffers: &'a [Buffer],
}

impl ExecutionContext<'_> {
    /// Get the underlying command buffer
    pub fn cmd(&mut self) -> &mut CommandBuffer { self.cmd }

    /// Get `sgpu::Image`
    pub fn image<A: ImageAccess>(&self, token: PassImage<A>) -> Image { self.images[token.index as usize] }

    /// Get `sgpu::ImageView`
    pub fn view<A: ImageAccess>(&self, token: PassImage<A>) -> ImageView { self.images[token.index as usize].default_view() }

    /// Get descriptor id of Storage image
    pub fn storage_id<A: StorageImageAccess>(&self, token: PassImage<A>) -> StorageImageId<A::U> { self.images[token.index as usize].default_view().storage_id() }

    /// Get descriptor id of Sampled image
    pub fn sampled_id<A: SampledImageAccess>(&self, token: PassImage<A>) -> SampledImageId { self.images[token.index as usize].default_view().sampled_id() }

    /// Get `sgpu::Buffer`
    pub fn buffer<A: BufferAccess>(&self, token: PassBuffer<A>) -> Buffer { self.buffers[token.index as usize] }

    /// Get descriptor id of Buffer
    pub fn descriptor_index<A: BufferAccess>(&self, token: PassBuffer<A>) -> BufferId<A::U> { self.buffers[token.index as usize].descriptor_index() }
}
