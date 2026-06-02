use std::{cell::RefCell, sync::Mutex};

use ash::vk;
use smallvec::smallvec;

use crate::{Counter, commands::*};

pub(crate) struct Queue {
    pub(crate) queue: vk::Queue,
    pub(crate) queue_type: QueueType,
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) cpu_signaled_value: Mutex<u64>, // mutex is okay as this enforeces the vulkan constraint of queue being single threaded
}

impl Queue {
    pub(crate) fn submit(&self, command_buffers: &[CommandBuffer]) -> Counter {
        // a submit scartch to reuse memory, idk if this is needed but here goes nothing ig
        struct SubmitScratch {
            waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
            cmd_infos: Vec<vk::CommandBufferSubmitInfo<'static>>,
        }

        impl SubmitScratch {
            fn clear(&mut self) {
                self.waits.clear();
                self.cmd_infos.clear();
            }
        }

        // do i even need thread local here
        thread_local! {
            static SUBMIT_SCRATCH: RefCell<SubmitScratch> = RefCell::new(SubmitScratch {
                waits: Vec::new(),
                cmd_infos: Vec::new(),
            });
        }

        let ctx = crate::CONTEXT.get().unwrap();

        // end all command buffers
        for cmd in command_buffers {
            unsafe {
                ctx.device.handle.end_command_buffer(cmd.handle).expect("vkEndCommandBuffer failed");
            }
        }

        // man do i even need a scratch..
        // hmm
        // submit is called so many times probs can just use it
        return SUBMIT_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.clear();

            for cmd in command_buffers {
                scratch.waits.extend_from_slice(&cmd.waits);
                scratch.cmd_infos.push(vk::CommandBufferSubmitInfo::default().command_buffer(cmd.handle));
            }

            let mut signal_value = self.cpu_signaled_value.lock().unwrap();
            *signal_value += 1;
            let counter = Counter::encode(self.queue_type, *signal_value);

            let signal_infos = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.semaphore)
                    .value(*signal_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS), // how can i chage this??
            ];

            let submit_infos = [
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&scratch.waits)
                    .command_buffer_infos(&scratch.cmd_infos)
                    .signal_semaphore_infos(&signal_infos),
            ];

            unsafe {
                ctx.device.handle.queue_submit2(self.queue, &submit_infos, vk::Fence::null()).expect("vkQueueSubmit2 failed");
            }

            counter
        });
    }

    pub(crate) fn poll(&self, value: u64) -> bool {
        let current_value = unsafe { crate::CONTEXT.get().unwrap().device.handle.get_semaphore_counter_value(self.semaphore).unwrap() };
        return current_value >= value;
    }
}

struct CommandPool {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    in_flight: usize,
    last_signal: Option<Counter>,
}

impl CommandPool {
    fn new(queue_type: QueueType) -> Self {
        let pool = crate::CONTEXT.get().expect("sgpu not initialized").device.create_command_pool(queue_type);

        return CommandPool {
            pool,
            buffers: Vec::new(),
            in_flight: 0,
            last_signal: None,
        };
    }

    fn reset(&mut self) {
        unsafe {
            crate::CONTEXT
                .get()
                .unwrap()
                .device
                .handle
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .expect("vkResetCommandPool failed");
        }

        self.in_flight = 0;
        self.last_signal = None;
    }

    fn allocate_buffer(&mut self) -> vk::CommandBuffer {
        let device = &crate::CONTEXT.get().unwrap().device.handle;

        if self.in_flight == self.buffers.len() {
            let new = unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(self.pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(8),
                    )
                    .expect("Failed to allocate Command buffer")
            };
            self.buffers.extend_from_slice(&new);
        }

        let buf = self.buffers[self.in_flight];
        self.in_flight += 1;

        unsafe {
            device
                .begin_command_buffer(buf, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .expect("Failed to begin command buffer");
        }

        return buf;
    }

    fn is_free(&self) -> bool {
        return match self.last_signal {
            None => self.in_flight == 0,
            Some(c) => crate::poll(c),
        };
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        crate::CONTEXT.get().unwrap().device.destroy_command_pool(self.pool);
    }
}

struct CommandPoolRing {
    head: usize,
    pools: Vec<CommandPool>,
    queue_type: QueueType,
}

impl CommandPoolRing {
    fn new(queue_type: QueueType, initial_size: usize) -> Self {
        Self {
            head: 0,
            pools: (0..initial_size).map(|_| CommandPool::new(queue_type)).collect(),
            queue_type,
        }
    }

    fn get(&mut self) -> (usize, vk::CommandBuffer) {
        let len = self.pools.len();

        for i in 0..len {
            let idx = (self.head + i) % len;

            if self.pools[idx].is_free() {
                if self.pools[idx].last_signal.is_some() {
                    self.pools[idx].reset();
                }

                self.head = idx;
                let buf = self.pools[idx].allocate_buffer();
                return (idx, buf);
            }
        }

        self.pools.push(CommandPool::new(self.queue_type));
        let idx = self.pools.len() - 1;
        let buf = self.pools[idx].allocate_buffer();

        return (idx, buf);
    }

    // just gets called by vkSubmit
    fn mark_submitted(&mut self, pool_idx: usize, counter: Counter) {
        self.pools[pool_idx].last_signal = Some(counter);
    }
}

pub(crate) struct ThreadCommandPools {
    rings: [CommandPoolRing; 3],
}

impl ThreadCommandPools {
    pub(crate) fn new() -> Self {
        Self {
            rings: std::array::from_fn(|i| CommandPoolRing::new(QueueType::QUEUE_TYPES[i], 3)),
        }
    }

    pub(crate) fn record(&mut self, queue_type: QueueType) -> CommandBuffer {
        let ring = &mut self.rings[queue_type as usize];
        let (pool_idx, handle) = ring.get();

        CommandBuffer {
            handle,
            queue: queue_type,
            pool_idx,
            waits: smallvec![],
        }
    }

    pub(crate) fn mark_submitted(&mut self, queue_type: QueueType, pool_idx: usize, counter: Counter) {
        self.rings[queue_type as usize].mark_submitted(pool_idx, counter);
    }
}
