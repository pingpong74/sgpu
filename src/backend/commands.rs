use std::{cell::Cell, sync::atomic::AtomicUsize};

use crate::{Counter, commands::QueueType};
use ash::vk;

// create a single ring buffer of command pools per thread
//
// keep track of which command pools are still in use, etc. that way we can reset them and ready them for reuse
struct CommandPool {
    pool: vk::CommandPool,
    last_signal: Option<Counter>,
    buffers: Vec<vk::CommandBuffer>,
    in_flight: usize,
}

impl CommandPool {
    fn new(queue_type: QueueType) -> CommandPool {
        let pool = crate::CONTEXT.get().unwrap().device.create_command_pool(queue_type);

        return CommandPool {
            pool,
            last_signal: None,
            buffers: Vec::new(),
            in_flight: 0,
        };
    }

    fn reset(&mut self) {
        unsafe {
            crate::CONTEXT.get().unwrap().device.handle.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()).unwrap();
        }

        self.in_flight = 0;
        self.last_signal = None;
    }

    fn allocate_buffer(&mut self) -> vk::CommandBuffer {
        let device = &crate::CONTEXT.get().unwrap().device.handle;

        if self.in_flight == self.buffers.len() {
            let alloc = unsafe {
                device
                    .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                        command_pool: self.pool,
                        level: vk::CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                        ..Default::default()
                    })
                    .unwrap()[0]
            };

            self.buffers.push(alloc);
        }

        let buf = self.buffers[self.in_flight];
        self.in_flight += 1;

        unsafe {
            device.begin_command_buffer(buf, &vk::CommandBufferBeginInfo::default()).unwrap();
        }

        return buf;
    }
}

struct CommandPoolRing {
    head: usize,
    pools: Vec<CommandPool>,
}

impl CommandPoolRing {
    fn new(queue_type: QueueType, num_of_pools: usize) -> Self {
        let mut pools = Vec::with_capacity(num_of_pools);

        for _ in 0..num_of_pools {
            pools.push(CommandPool::new(queue_type));
        }

        Self {
            head: 0,
            pools,
        }
    }

    fn get(&mut self) -> &mut CommandPool {
        let len = self.pools.len();

        for _ in 0..len {
            let idx = self.head;
            self.head = (self.head + 1) % len;

            let ready = {
                let pool = &self.pools[idx];

                match pool.last_signal {
                    None => true,
                    Some(counter) => crate::poll(counter),
                }
            };

            if ready {
                let pool = &mut self.pools[idx];
                pool.in_flight = 0;

                return pool;
            }
        }

        // fallback: all busy → wait oldest
        let idx = self.head;

        if let Some(counter) = self.pools[idx].last_signal {
            crate::wait(counter);
        }

        self.pools[idx].reset();

        &mut self.pools[idx]
    }
}

pub(crate) struct ThreadCommandPools {
    thread_id: usize,
    pools: [CommandPoolRing; 3],
}

impl ThreadCommandPools {
    // TODO: Make the number of pools per thread per queue a parameter?
    pub(crate) fn new(id: usize) -> ThreadCommandPools {
        return ThreadCommandPools {
            pools: std::array::from_fn(|i| CommandPoolRing::new(QueueType::QUEUE_TYPES[i], 5)),
            thread_id: id,
        };
    }
}
