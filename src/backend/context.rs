use crate::{
    Buffer, BufferDescription, Counter,
    api::SgpuInititizationInfo,
    backend::{
        commands::*,
        device::{Device, Queue},
        resources::InnerBuffer,
    },
    commands::QueueType,
};

use super::instance::Instance;
use slotmap::{DefaultKey, SlotMap};

use std::{
    cell::UnsafeCell,
    sync::{RwLock, atomic::AtomicUsize},
};

pub(crate) struct Context {
    instance: Instance,
    pub(crate) device: Device,

    queues: [Queue; 3],

    buffers: UnsafeCell<SlotMap<DefaultKey, InnerBuffer>>,

    pub(crate) thread_cmd_pool_pool: RwLock<Vec<ThreadCommandPools>>,
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

            thread_cmd_pool_pool: RwLock::new(vec![]),
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
}

impl Drop for Context {
    fn drop(&mut self) {
        self.instance.cleanup();
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
