use crate::{Buffer, Counter};
use ash::vk;
use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
}

impl QueueType {
    pub(crate) const QUEUE_TYPES: [QueueType; 3] = [
        QueueType::Graphics,
        QueueType::Compute,
        QueueType::Transfer,
    ];
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PipelineStage: u64 {
        const NONE = 0;

        const TOP_OF_PIPE = vk::PipelineStageFlags2::TOP_OF_PIPE.as_raw();
        const BOTTOM_OF_PIPE = vk::PipelineStageFlags2::BOTTOM_OF_PIPE.as_raw();
        const DRAW_INDIRECT = vk::PipelineStageFlags2::DRAW_INDIRECT.as_raw();
        const VERTEX_INPUT = vk::PipelineStageFlags2::VERTEX_INPUT.as_raw();
        const VERTEX_SHADER = vk::PipelineStageFlags2::VERTEX_SHADER.as_raw();
        const TESSELLATION_CONTROL_SHADER = vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER.as_raw();
        const TESSELLATION_EVALUATION_SHADER = vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER.as_raw();
        const GEOMETRY_SHADER = vk::PipelineStageFlags2::GEOMETRY_SHADER.as_raw();
        const FRAGMENT_SHADER = vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw();
        const EARLY_FRAGMENT_TESTS = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw();
        const LATE_FRAGMENT_TESTS = vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw();
        const COLOR_ATTACHMENT_OUTPUT = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw();
        const COMPUTE_SHADER = vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw();

        const ALL_TRANSFER = vk::PipelineStageFlags2::ALL_TRANSFER.as_raw();
        const TRANSFER = vk::PipelineStageFlags2::TRANSFER.as_raw();
        const COPY = vk::PipelineStageFlags2::COPY.as_raw();
        const RESOLVE = vk::PipelineStageFlags2::RESOLVE.as_raw();
        const BLIT = vk::PipelineStageFlags2::BLIT.as_raw();
        const CLEAR = vk::PipelineStageFlags2::CLEAR.as_raw();

        const RAY_TRACING_SHADER = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw();
        const ACCELERATION_STRUCTURE_BUILD = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR.as_raw();
        const ACCELERATION_STRUCTURE_COPY = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_COPY_KHR.as_raw();

        const HOST = vk::PipelineStageFlags2::HOST.as_raw();
        const ALL_GRAPHICS = vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw();
        const ALL_COMMANDS = vk::PipelineStageFlags2::ALL_COMMANDS.as_raw();
    }
}

impl PipelineStage {
    #[inline]
    pub const fn to_vk(self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::from_raw(self.bits())
    }
}

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) queue: QueueType,
    pub(crate) pool_idx: usize,
    pub(crate) waits: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 3]>,
}

impl CommandBuffer {
    pub fn fill_buffer(&mut self, buffer: &Buffer, offset: u64, size: u64, data: u32) {
        let ctx = crate::CONTEXT.get().unwrap();

        let buffer = ctx.get_buffer_inner(buffer);

        unsafe {
            ctx.device.handle.cmd_fill_buffer(self.handle, buffer.buffer, offset, size, data);
        }
    }

    pub fn wait_for(&mut self, counter: Counter, stage: PipelineStage) {
        let (queue, value) = counter.decode();

        if queue == self.queue {
            return;
        }

        let ctx = crate::CONTEXT.get().expect("sgpu not initialized");
        let semaphore = ctx.queues[queue as usize].semaphore;

        self.waits.push(vk::SemaphoreSubmitInfo {
            semaphore,
            value,
            device_index: 0,
            stage_mask: stage.to_vk(),
            ..Default::default()
        });
    }
}
