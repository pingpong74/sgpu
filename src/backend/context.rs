use crate::{
    Buffer, BufferDescription, Counter, Image, ImageDescription, ImageView,
    api::SgpuInititizationInfo,
    backend::{
        commands::*,
        device::Device,
        resources::{InnerBuffer, InnerImage},
    },
    commands::{CommandBuffer, QueueType},
};

use super::instance::Instance;
use slotmap::{DefaultKey, SlotMap};

use std::{collections::HashMap, sync::RwLock, thread::ThreadId};

pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) device: Device,

    pub(crate) queues: [Queue; 3],

    pub(crate) buffers: RwLock<SlotMap<DefaultKey, InnerBuffer>>,
    pub(crate) images: RwLock<SlotMap<DefaultKey, InnerImage>>,

    pub(crate) thread_cmd_pool_pool: RwLock<HashMap<ThreadId, ThreadCommandPools>>,
}

impl Context {
    pub(crate) fn new(init_info: &SgpuInititizationInfo) -> Context {
        let instance = Instance::new(init_info);
        let device = Device::new(&init_info, &instance);
        let queues = device.get_queues();

        return Context {
            instance: instance,
            device: device,
            queues: queues,

            buffers: RwLock::new(Default::default()),
            images: RwLock::new(Default::default()),

            thread_cmd_pool_pool: RwLock::new(HashMap::new()),
        };
    }

    pub(crate) fn create_buffer(&self, desc: &BufferDescription) -> Buffer {
        let inner_buffer = self.device.create_buffer(desc);

        return Buffer {
            id: self.buffers.write().unwrap().insert(inner_buffer),
        };
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
        let view = inner_image.image_views[0].view;
        let id = self.images.write().unwrap().insert(inner_image);

        return Image {
            default_view: ImageView {
                raw: view,
                image_key: id,
                id: 0,
            },
            id: id,
        };
    }

    pub(crate) fn destroy_image(&self, image: Image) {
        let inner = self.images.write().unwrap().remove(image.id).expect("Tried to destroy Image twice");

        self.device.destroy_image(inner);
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

        self.instance.cleanup();
        self.device.cleanup();
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
