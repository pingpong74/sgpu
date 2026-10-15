use sgpu::{AccessType, Buffer, BufferId, CommandBuffer, Image, ImageView, QueueType, SampledImageId, StorageImageId};

use crate::access::{BufferAccess, ImageAccess, SampledImageAccess, StorageImageAccess};
use crate::resource::{Id, PassBuffer, PassImage};

pub trait Node<R, D> {
    const QUEUE: QueueType;
    type Config;

    fn declare(resources: &R, config: &Self::Config, write: &mut NodeWriter) -> Self;

    fn execute(&self, data: &D, context: &mut ExecutionContext<'_>);
}

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

    pub fn image<A: ImageAccess>(&mut self, id: Id<Image>) -> PassImage<A> {
        assert!((id.index as usize) < self.image_count, "image id {id:?} does not belong to this graph (foreign or stale Id?)");
        self.images.push((id.index, A::ACCESS));
        PassImage::new(id.index)
    }

    pub fn buffer<A: BufferAccess>(&mut self, id: Id<Buffer>) -> PassBuffer<A> {
        assert!((id.index as usize) < self.buffer_count, "buffer id {id:?} does not belong to this graph (foreign or stale Id?)");
        self.buffers.push((id.index, A::ACCESS));
        PassBuffer::new(id.index)
    }
}

pub struct ExecutionContext<'a> {
    pub(crate) cmd: &'a mut CommandBuffer,
    pub(crate) images: &'a [Image],
    pub(crate) buffers: &'a [Buffer],
}

impl ExecutionContext<'_> {
    #[inline]
    pub fn cmd(&mut self) -> &mut CommandBuffer { self.cmd }

    pub fn image<A: ImageAccess>(&self, token: PassImage<A>) -> Image { self.images[token.index as usize] }

    pub fn view<A: ImageAccess>(&self, token: PassImage<A>) -> ImageView { self.images[token.index as usize].default_view() }

    pub fn storage_id<A: StorageImageAccess>(&self, token: PassImage<A>) -> StorageImageId<A::U> { self.images[token.index as usize].default_view().storage_id() }

    pub fn sampled_id<A: SampledImageAccess>(&self, token: PassImage<A>) -> SampledImageId { self.images[token.index as usize].default_view().sampled_id() }

    pub fn buffer<A: BufferAccess>(&self, token: PassBuffer<A>) -> Buffer { self.buffers[token.index as usize] }

    pub fn descriptor_index<A: BufferAccess>(&self, token: PassBuffer<A>) -> BufferId<A::U> { self.buffers[token.index as usize].descriptor_index() }
}
