use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, ImageView, ImageViewDescription, QueueType, SampledImageId, StorageImageId};

use crate::access::{self, BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::resource::{Id, ImageViewPool, InnerBuffer, InnerImage, PassBuffer, PassImage, PassView};

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

    type Config = Id<Image>;

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
    graph_id: u32,
    images: &'a [InnerImage],
    buffers: &'a [InnerBuffer],
    views: &'a mut ImageViewPool,
    pub(crate) image_accesses: Vec<(u32, AccessType)>,
    pub(crate) buffer_accesses: Vec<(u32, AccessType)>,
}

impl<'a> NodeWriter<'a> {
    pub(crate) fn new(graph_id: u32, images: &'a [InnerImage], buffers: &'a [InnerBuffer], views: &'a mut ImageViewPool) -> Self {
        NodeWriter {
            graph_id,
            images,
            buffers,
            views,
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

    /// Get an `sgpu::ImageView` of the Image with given Access
    pub fn view<A: ImageAccess>(&mut self, id: Id<Image>, desc: &ImageViewDescription) -> PassView<A> {
        let index = id.check(self.graph_id, self.images.len());
        self.image_accesses.push((index, A::ACCESS));
        PassView::new(self.views.register(index, *desc))
    }
}

/// Consists of a CommandBuffer
/// Used to get `sgpu::Image` and `sgpu::Buffer`
pub struct ExecutionContext<'a> {
    pub(crate) cmd: &'a mut CommandBuffer,
    pub(crate) images: &'a [InnerImage],
    pub(crate) buffers: &'a [InnerBuffer],
    pub(crate) views: &'a ImageViewPool,
    pub(crate) curr_frame: usize,
}

impl ExecutionContext<'_> {
    /// Get the underlying command buffer
    pub fn cmd(&mut self) -> &mut CommandBuffer { self.cmd }

    /// Get `sgpu::Image`
    pub fn image<A: ImageAccess>(&self, image: PassImage<A>) -> Image { self.images[image.index as usize].image_at(self.curr_frame) }

    /// Get the `sgpu::ImageView`
    pub fn view<A: ImageAccess>(&self, image_view: PassView<A>) -> ImageView { self.views.get(image_view.index, self.images, self.curr_frame) }

    /// Get descriptor id of Storage image
    pub fn storage_id<A: StorageImageAccess>(&self, image: PassImage<A>) -> StorageImageId<A::U> { self.images[image.index as usize].image_at(self.curr_frame).default_view().storage_id() }

    /// Get descriptor id of Sampled image
    pub fn sampled_id<A: SampledImageAccess>(&self, image: PassImage<A>) -> SampledImageId { self.images[image.index as usize].image_at(self.curr_frame).default_view().sampled_id() }

    /// Get `sgpu::Buffer`
    pub fn buffer<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> Buffer { self.buffers[buffer.index as usize].buffer_at(self.curr_frame) }

    /// Get descriptor id of Buffer
    pub fn descriptor_index<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> BufferId<A::U> { self.buffers[buffer.index as usize].buffer_at(self.curr_frame).descriptor_index() }
}
