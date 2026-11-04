use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, ImageView, ImageViewDescription, QueueType, SampledImageId, StorageImageId};

use crate::access::{self, BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::graph::ResourcePool;
use crate::resource::{PassBuffer, PassBufferId, PassImage, PassImageId, PassView};

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

pub(crate) struct PresentNode {
    pub(crate) _swapchain_img: PassImage<access::Present>,
}

impl<R, D> Node<R, D> for PresentNode {
    const QUEUE: QueueType = QueueType::Graphics;

    type Config = PassImageId;

    fn declare(_: &R, config: &Self::Config, write: &mut NodeWriter) -> Self {
        Self {
            _swapchain_img: write.image(*config),
        }
    }

    fn execute(&self, _: &D, _: &mut ExecutionContext<'_>) {
        // do nothing, just mark this node as a present
    }
}

/// Used to register the access's of resources
pub struct NodeWriter<'a> {
    pool: &'a mut ResourcePool,
    pub(crate) image_accesses: Vec<(PassImageId, AccessType)>,
    pub(crate) buffer_accesses: Vec<(PassBufferId, AccessType)>,
}

impl<'a> NodeWriter<'a> {
    pub(crate) fn new(pool: &'a mut ResourcePool) -> Self {
        NodeWriter {
            pool,
            image_accesses: Vec::new(),
            buffer_accesses: Vec::new(),
        }
    }

    /// Get Image with given Access
    pub fn image<A: ImageAccess>(&mut self, id: PassImageId) -> PassImage<A> {
        self.image_accesses.push((id, A::ACCESS));
        PassImage::new(self.pool.check_image_raw(id))
    }

    /// Get Buffer with given Access
    pub fn buffer<A: BufferAccess>(&mut self, id: PassBufferId) -> PassBuffer<A> {
        self.buffer_accesses.push((id, A::ACCESS));
        PassBuffer::new(self.pool.check_buffer_raw(id))
    }

    /// Get an `sgpu::ImageView` of the Image with given Access
    pub fn view<A: ImageAccess>(&mut self, id: PassImageId, desc: &ImageViewDescription) -> PassView<A> {
        self.image_accesses.push((id, A::ACCESS));
        PassView::new(self.pool.register_view(self.pool.check_image_raw(id), *desc))
    }
}

/// Consists of a CommandBuffer
/// Used to get `sgpu::Image` and `sgpu::Buffer`
pub struct ExecutionContext<'a> {
    pub(crate) cmd: &'a mut CommandBuffer,
    pub(crate) pool: &'a ResourcePool,
    pub(crate) curr_frame: usize,
}

impl<'a> ExecutionContext<'a> {
    /// Get the underlying command buffer
    pub fn cmd(&mut self) -> &mut CommandBuffer { self.cmd }

    /// Get `sgpu::Image`
    pub fn image<A: ImageAccess>(&self, image: PassImage<A>) -> Image { self.pool.image_at(image.index, self.curr_frame) }

    /// Get the `sgpu::ImageView`
    pub fn view<A: ImageAccess>(&self, image_view: PassView<A>) -> ImageView { self.pool.view(image_view.index, self.curr_frame) }

    /// Get descriptor id of Storage image
    pub fn storage_id<A: StorageImageAccess>(&self, image: PassImage<A>) -> StorageImageId<A::U> { self.pool.image_at(image.index, self.curr_frame).default_view().storage_id() }

    /// Get descriptor id of Sampled image
    pub fn sampled_id<A: SampledImageAccess>(&self, image: PassImage<A>) -> SampledImageId { self.pool.image_at(image.index, self.curr_frame).default_view().sampled_id() }

    /// Get `sgpu::Buffer`
    pub fn buffer<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> Buffer { self.pool.buffer_at(buffer.index, self.curr_frame) }

    /// Get descriptor id of Buffer
    pub fn descriptor_index<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> BufferId<A::U> { self.pool.buffer_at(buffer.index, self.curr_frame).descriptor_index() }
}
