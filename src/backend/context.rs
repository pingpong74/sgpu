use crate::{
    Buffer, BufferDescription, Counter, Image, ImageDescription, ImageView, ImageViewDescription,
    api::SgpuInititizationInfo,
    backend::{
        commands::*,
        descriptors::BindlessDescriptorSet,
        device::Device,
        resources::{InnerBuffer, InnerImage, InnerImageView},
    },
    commands::{CommandBuffer, QueueType},
    pipeline::{RasterizationPipeline, RasterizationPipelineDescription},
};

use super::instance::Instance;
use slotmap::{DefaultKey, Key, SlotMap};

use std::{collections::HashMap, sync::RwLock, thread::ThreadId};

pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) bindless_descriptor_set: BindlessDescriptorSet,

    pub(crate) queues: [Queue; 3],

    pub(crate) buffers: RwLock<SlotMap<DefaultKey, InnerBuffer>>,
    pub(crate) images: RwLock<SlotMap<DefaultKey, InnerImage>>,
    pub(crate) image_views: RwLock<SlotMap<DefaultKey, InnerImageView>>,

    pub(crate) thread_cmd_pool_pool: RwLock<HashMap<ThreadId, ThreadCommandPools>>,
}

impl Context {
    pub(crate) fn new(init_info: &SgpuInititizationInfo) -> Context {
        let instance = Instance::new(init_info);
        let device = Device::new(&init_info, &instance);
        let queues = device.get_queues();
        let bindless_set = BindlessDescriptorSet::new(&device);

        return Context {
            instance: instance,
            device: device,
            queues: queues,
            bindless_descriptor_set: bindless_set,

            buffers: RwLock::new(Default::default()),
            images: RwLock::new(Default::default()),
            image_views: RwLock::new(Default::default()),

            thread_cmd_pool_pool: RwLock::new(HashMap::new()),
        };
    }

    pub(crate) fn create_buffer(&self, desc: &BufferDescription) -> Buffer {
        let inner_buffer = self.device.create_buffer(desc);
        let buffer = inner_buffer.buffer;
        let id = self.buffers.write().unwrap().insert(inner_buffer);
        // Janky stuff
        self.bindless_descriptor_set.write_buffer(&self.device, buffer, id.data().as_ffi() as u32);

        return Buffer { id: id };
    }

    pub(crate) unsafe fn get_buffer_inner(&self, id: &Buffer) -> &InnerBuffer {
        let ptr = self.buffers.read().unwrap().get(id.id).expect("Invalid Buffer") as *const InnerBuffer;
        return unsafe { &*ptr };
    }

    pub(crate) fn destroy_buffer(&self, buffer: Buffer) {
        let inner = self.buffers.write().unwrap().remove(buffer.id).expect("Tried destroy buffer twice");

        self.device.destroy_buffer(inner);
    }

    pub(crate) fn create_image(&self, desc: &ImageDescription) -> Image {
        let inner_image = self.device.create_image(desc);
        let view = self.device.create_image_view(&inner_image, &desc.default_view);
        let raw = view.view;
        let id = self.images.write().unwrap().insert(inner_image);

        self.bindless_descriptor_set.write_sampled_image(&self.device, view.view, id.data().as_ffi() as u32);
        self.bindless_descriptor_set.write_storage_image(&self.device, view.view, id.data().as_ffi() as u32);

        return Image {
            default_view: ImageView {
                raw: raw,
                id: self.image_views.write().unwrap().insert(view),
            },
            id: id,
        };
    }

    pub(crate) fn create_image_view(&self, desc: &ImageViewDescription, image: Image) -> ImageView {
        let inner_view = self.device.create_image_view(self.images.read().unwrap().get(image.id).expect("Invalid image Id"), desc);

        return ImageView {
            raw: inner_view.view,
            id: self.image_views.write().unwrap().insert(inner_view),
        };
    }

    pub(crate) fn destroy_image(&self, image: Image) {
        let inner = self.images.write().unwrap().remove(image.id).expect("Tried to destroy Image twice");

        self.device.destroy_image(inner);
    }

    pub(crate) fn create_raster_pipeline(&self, dec: &RasterizationPipelineDescription) -> RasterizationPipeline {
        return self.device.create_raster_pipeline(&self.bindless_descriptor_set, dec);
    }

    pub(crate) fn poll(&self, counter: Counter) -> bool {
        let (queue_type, value) = counter.decode();

        return self.queues[queue_type as usize].poll(value);
    }

    pub(crate) fn wait_idle(&self) {
        self.device.wait_idle();
    }

    pub(crate) fn wait(&self, counter: Counter) {
        let (queue_type, value) = counter.decode();
        self.device.wait_queue(&self.queues[queue_type as usize], value);
    }

    pub(crate) fn record(&self, queue_type: QueueType) -> CommandBuffer {
        let thread_id = std::thread::current().id();
        return self.thread_cmd_pool_pool.write().unwrap().entry(thread_id).or_insert_with(ThreadCommandPools::new).record(queue_type);
    }

    pub(crate) fn submit(&self, command_buffers: &[CommandBuffer]) -> Counter {
        let queue_type = command_buffers[0].queue;

        debug_assert!(command_buffers.iter().all(|c| c.queue == queue_type), "All command buffers must be from the same queue");

        let counter = self.queues[queue_type as usize].submit(command_buffers);

        // Mark all the pools to be free
        let thread_id = std::thread::current().id();
        let mut pools = self.thread_cmd_pool_pool.write().unwrap();
        if let Some(thread_pools) = pools.get_mut(&thread_id) {
            for cmd in command_buffers {
                thread_pools.mark_submitted(queue_type, cmd.pool_idx, counter);
            }
        }

        return counter;
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.buffers.write().unwrap().drain().for_each(|(_, buffer)| self.device.destroy_buffer(buffer));
        self.images.write().unwrap().drain().for_each(|(_, img)| self.device.destroy_image(img));

        self.bindless_descriptor_set.cleanup(&self.device);
        self.device.cleanup();
        self.instance.cleanup();
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
