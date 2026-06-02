use crate::{
    Buffer, BufferDescription, Counter,
    api::SgpuInititizationInfo,
    backend::{commands::*, device::Device, resources::InnerBuffer},
    commands::{CommandBuffer, QueueType},
};

use super::instance::Instance;
use slotmap::{DefaultKey, SlotMap};

use std::{cell::UnsafeCell, collections::HashMap, sync::RwLock, thread::ThreadId};

pub(crate) struct Context {
    instance: Instance,
    pub(crate) device: Device,

    pub(crate) queues: [Queue; 3],

    buffers: UnsafeCell<SlotMap<DefaultKey, InnerBuffer>>,

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

            buffers: UnsafeCell::new(Default::default()),

            thread_cmd_pool_pool: RwLock::new(HashMap::new()),
        };
    }

    pub(crate) fn create_buffer(&self, desc: &BufferDescription) -> Buffer {
        let inner_buffer = self.device.create_buffer(desc);

        return unsafe {
            Buffer {
                id: self.buffers.get().as_mut_unchecked().insert(inner_buffer),
            }
        };
    }

    pub(crate) fn get_buffer_inner(&self, id: &Buffer) -> &InnerBuffer {
        return unsafe { self.buffers.get().as_mut_unchecked().get(id.id).expect("Inavild Buffer") };
    }

    pub(crate) fn destroy_buffer(&self, buffer: Buffer) {
        let inner = unsafe { self.buffers.get().as_mut_unchecked().remove(buffer.id).expect("Tried destroy buffer twice") };

        self.device.destroy_buffer(inner);
    }

    pub(crate) fn poll(&self, counter: Counter) -> bool {
        let (queue_type, value) = counter.decode();

        return self.queues[queue_type as usize].poll(value);
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
        unsafe {
            self.buffers.get().as_mut_unchecked().drain().for_each(|(_, buffer)| self.device.destroy_buffer(buffer));
        }

        self.instance.cleanup();
        self.device.cleanup();
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
