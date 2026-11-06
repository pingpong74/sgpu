use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, ImageView, ImageViewDescription, QueueType, SampledImageId, StorageImageId};

use crate::access::{self, BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::graph::ResourcePool;
use crate::resource::{ImageViewRef, PassBuffer, PassBufferId, PassImage, PassImageId, PassImageView};

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
    pub(crate) _swapchain_img: PassImageView<access::Present>,
}

impl<R, D> Node<R, D> for PresentNode {
    const QUEUE: QueueType = QueueType::Graphics;

    type Config = PassImageId;

    fn declare(_: &R, config: &Self::Config, write: &mut NodeWriter) -> Self {
        Self {
            _swapchain_img: write.default_view(*config),
        }
    }

    fn execute(&self, _: &D, _: &mut ExecutionContext<'_>) {
        // do nothing, just mark this node as a present
    }
}

/// One image use, including the view whose subresources it accesses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageUse {
    pub(crate) image: PassImageId,
    pub(crate) view: ImageViewRef,
    pub(crate) access: AccessType,
}

/// Used to register the access's of resources
pub struct NodeWriter<'a> {
    pool: &'a mut ResourcePool,
    pub(crate) image_accesses: Vec<ImageUse>,
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

    /// Get Buffer with given Access
    pub fn buffer<A: BufferAccess>(&mut self, id: PassBufferId) -> PassBuffer<A> {
        self.buffer_accesses.push((id, A::ACCESS));
        PassBuffer::new(self.pool.check_buffer_raw(id))
    }

    /// Get Image with given Access
    pub fn image<A: ImageAccess>(&mut self, id: PassImageId) -> PassImage<A> {
        self.image_accesses.push(ImageUse {
            image: id,
            view: ImageViewRef::Default,
            access: A::ACCESS,
        });
        PassImage::new(self.pool.check_image_raw(id))
    }

    /// Get an `sgpu::ImageView` of the Image with given Access
    pub fn view<A: ImageAccess>(&mut self, id: PassImageId, desc: &ImageViewDescription) -> PassImageView<A> {
        let view = self.pool.register_view(self.pool.check_image_raw(id), *desc);
        self.image_accesses.push(ImageUse {
            image: id,
            view: ImageViewRef::Registered(view),
            access: A::ACCESS,
        });
        PassImageView::new(view, self.pool.check_image_raw(id))
    }

    /// Get an `sgpu::ImageView` of the Image with given Access
    pub fn default_view<A: ImageAccess>(&mut self, id: PassImageId) -> PassImageView<A> {
        self.image_accesses.push(ImageUse {
            image: id,
            view: ImageViewRef::Default,
            access: A::ACCESS,
        });
        PassImageView::default_view(self.pool.check_image_raw(id))
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

    /// Get `sgpu::Image`
    pub fn image_from_view<A: ImageAccess>(&self, view: PassImageView<A>) -> Image { self.pool.image_at(view.image, self.curr_frame) }

    /// Get the `sgpu::ImageView`
    pub fn view<A: ImageAccess>(&self, view: PassImageView<A>) -> ImageView {
        match view.get_index() {
            Some(i) => self.pool.view(i, self.curr_frame),
            None => self.pool.image_at(view.image, self.curr_frame).default_view(),
        }
    }

    /// Get descriptor id of Storage image
    pub fn storage_id<A: StorageImageAccess>(&self, view: PassImageView<A>) -> StorageImageId<A::U> {
        match view.get_index() {
            Some(i) => self.pool.view(i, self.curr_frame).storage_id(),
            None => self.pool.image_at(view.image, self.curr_frame).default_view().storage_id(),
        }
    }

    /// Get descriptor id of Sampled image
    pub fn sampled_id<A: SampledImageAccess>(&self, view: PassImageView<A>) -> SampledImageId {
        match view.get_index() {
            Some(i) => self.pool.view(i, self.curr_frame).sampled_id(),
            None => self.pool.image_at(view.image, self.curr_frame).default_view().sampled_id(),
        }
    }

    /// Get `sgpu::Buffer`
    pub fn buffer<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> Buffer { self.pool.buffer_at(buffer.index, self.curr_frame) }

    /// Get descriptor id of Buffer
    pub fn descriptor_index<A: BufferAccess>(&self, buffer: PassBuffer<A>) -> BufferId<A::U> { self.pool.buffer_at(buffer.index, self.curr_frame).descriptor_index() }
}
