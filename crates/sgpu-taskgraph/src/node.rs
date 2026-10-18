use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, QueueType, SampledImageId, StorageImageId};

use crate::access::{BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::resource::{Id, InnerImage, PassBuffer, PassImage};

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
pub struct NodeWriter<'a> {
    graph_id: u32,
    images: &'a [InnerImage],
    buffers: &'a [Buffer],
    pub(crate) image_accesses: Vec<(u32, AccessType)>,
    pub(crate) buffer_accesses: Vec<(u32, AccessType)>,
}

impl<'a> NodeWriter<'a> {
    pub(crate) fn new(graph_id: u32, images: &'a [InnerImage], buffers: &'a [Buffer]) -> Self {
        NodeWriter {
            graph_id,
            images,
            buffers,
            image_accesses: Vec::new(),
            buffer_accesses: Vec::new(),
        }
    }

    /// Get Image with given Access
    pub fn image<A: ImageAccess>(&mut self, id: Id<Image>) -> PassImage<A> {
        let index = id.check(self.graph_id, self.images.len());
        self.image_accesses.push((index, A::ACCESS));
        PassImage::new(index)
    }

    /// Get Buffer with given Access
    pub fn buffer<A: BufferAccess>(&mut self, id: Id<Buffer>) -> PassBuffer<A> {
        let index = id.check(self.graph_id, self.buffers.len());
        self.buffer_accesses.push((index, A::ACCESS));
        PassBuffer::new(index)
    }
}

/// Consists of a CommandBuffer
/// Used to get `sgpu::Image` and `sgpu::Buffer`
pub struct ExecutionContext<'a> {
    pub(crate) cmd: &'a mut CommandBuffer,
    pub(crate) images: &'a [InnerImage],
    pub(crate) buffers: &'a [Buffer],
}

impl ExecutionContext<'_> {
    /// Get the underlying command buffer
    pub fn cmd(&mut self) -> &mut CommandBuffer { self.cmd }

    /// Get `sgpu::Image`
    pub fn image<A: ImageAccess>(&self, token: PassImage<A>) -> Image { self.images[token.index as usize].image() }

    /// Get descriptor id of Storage image
    pub fn storage_id<A: StorageImageAccess>(&self, token: PassImage<A>) -> StorageImageId<A::U> { self.images[token.index as usize].image().default_view().storage_id() }

    /// Get descriptor id of Sampled image
    pub fn sampled_id<A: SampledImageAccess>(&self, token: PassImage<A>) -> SampledImageId { self.images[token.index as usize].image().default_view().sampled_id() }

    /// Get `sgpu::Buffer`
    pub fn buffer<A: BufferAccess>(&self, token: PassBuffer<A>) -> Buffer { self.buffers[token.index as usize] }

    /// Get descriptor id of Buffer
    pub fn descriptor_index<A: BufferAccess>(&self, token: PassBuffer<A>) -> BufferId<A::U> { self.buffers[token.index as usize].descriptor_index() }
}
