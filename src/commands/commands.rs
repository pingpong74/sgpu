use crate::{commands::rendering_recorder::*, swapchain::AcquiredImage, *};
use ash::vk;
use slotmap::{DefaultKey, Key};
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

// Render begin info
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderArea {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

impl LoadOp {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::AttachmentLoadOp {
        match self {
            Self::Load => vk::AttachmentLoadOp::LOAD,
            Self::Clear => vk::AttachmentLoadOp::CLEAR,
            Self::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoreOp {
    Store,
    DontCare,
    None,
}

impl StoreOp {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::AttachmentStoreOp {
        match self {
            Self::Store => vk::AttachmentStoreOp::STORE,
            Self::DontCare => vk::AttachmentStoreOp::DONT_CARE,
            Self::None => vk::AttachmentStoreOp::NONE,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResolveMode {
    None,
    SampleZero,
    Average,
    Min,
    Max,
}

impl ResolveMode {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::ResolveModeFlags {
        match self {
            Self::None => vk::ResolveModeFlags::NONE,
            Self::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            Self::Average => vk::ResolveModeFlags::AVERAGE,
            Self::Min => vk::ResolveModeFlags::MIN,
            Self::Max => vk::ResolveModeFlags::MAX,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClearValue {
    ColorFloat([f32; 4]),
    ColorInt([i32; 4]),
    ColorUint([u32; 4]),
    DepthStencil {
        depth: f32,
        stencil: u32,
    },
}

impl ClearValue {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::ClearValue {
        match self {
            Self::ColorFloat(v) => vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: *v,
                },
            },
            Self::ColorInt(v) => vk::ClearValue {
                color: vk::ClearColorValue { int32: *v },
            },
            Self::ColorUint(v) => vk::ClearValue {
                color: vk::ClearColorValue { uint32: *v },
            },
            Self::DepthStencil {
                depth,
                stencil,
            } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: *depth,
                    stencil: *stencil,
                },
            },
        }
    }

    /// Common helper for zero clear
    pub const fn black() -> Self {
        Self::ColorFloat([0.0, 0.0, 0.0, 1.0])
    }

    /// Common helper for white clear
    pub const fn white() -> Self {
        Self::ColorFloat([1.0, 1.0, 1.0, 1.0])
    }

    /// Common helper for depth clear
    pub const fn depth_one() -> Self {
        Self::DepthStencil {
            depth: 1.0,
            stencil: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct RenderingAttachment {
    pub image_view: ImageView,
    pub resolve_mode: ResolveMode,
    pub resolve_image_view: Option<ImageView>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: ClearValue,
}

impl Default for RenderingAttachment {
    fn default() -> Self {
        Self {
            image_view: ImageView {
                raw: ash::vk::ImageView::null(),
                image_key: DefaultKey::null(),
                id: 0,
            },
            resolve_image_view: None,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            resolve_mode: ResolveMode::None,
            clear_value: ClearValue::ColorFloat([0.0, 0.0, 0.0, 0.0]),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RenderingFlags: u32 {
        const NONE = 0;
        const CONTENTS_SECONDARY_COMMAND_BUFFERS = vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS.as_raw();
        const SUSPENDING = vk::RenderingFlags::SUSPENDING.as_raw();
        const RESUMING = vk::RenderingFlags::RESUMING.as_raw();
    }
}

impl RenderingFlags {
    pub const fn to_vk(self) -> vk::RenderingFlags {
        vk::RenderingFlags::from_raw(self.bits())
    }
}
pub struct RenderingBeginInfo<'a> {
    pub render_area: RenderArea,
    pub rendering_flags: RenderingFlags,
    pub view_mask: u32,
    pub layer_count: u32,
    pub color_attachments: &'a [RenderingAttachment],
    pub depth_attachment: Option<RenderingAttachment>,
    pub stencil_attachment: Option<RenderingAttachment>,
}

impl<'a> Default for RenderingBeginInfo<'a> {
    fn default() -> Self {
        Self {
            render_area: RenderArea {
                offset: Offset2D { x: 0, y: 0 },
                extent: Extent2D {
                    width: 0,
                    height: 0,
                },
            },
            rendering_flags: RenderingFlags::NONE,
            view_mask: 0,
            layer_count: 1,
            color_attachments: &[],
            depth_attachment: None,
            stencil_attachment: None,
        }
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

        unsafe {
            let buffer = ctx.get_buffer_inner(buffer);
            ctx.device.handle.cmd_fill_buffer(self.handle, buffer.buffer, offset, size, data);
        }
    }

    pub fn wait_for_swapchain_image(&mut self, image: &AcquiredImage) {
        self.waits.push(vk::SemaphoreSubmitInfo {
            semaphore: image.acquire_semaphore,
            value: 0,
            device_index: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        });
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

    // TODO: use a scratch?
    // give seperate struct or smt man
    pub fn begin_rendering<F: FnOnce(RenderRecorder)>(&mut self, rendering_begin_info: &RenderingBeginInfo, f: F) {
        let ctx = crate::CONTEXT.get().expect("sgpu not initialized");
        let mut color_attachment_info = SmallVec::<[vk::RenderingAttachmentInfo; 3]>::new();

        for color_attachement in rendering_begin_info.color_attachments {
            color_attachment_info.push(
                vk::RenderingAttachmentInfo::default()
                    .resolve_mode(color_attachement.resolve_mode.to_vk())
                    .image_view(color_attachement.image_view.raw)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .resolve_image_view(color_attachement.resolve_image_view.map(|i| i.raw).unwrap_or(vk::ImageView::null()))
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(color_attachement.load_op.to_vk())
                    .store_op(color_attachement.store_op.to_vk())
                    .clear_value(color_attachement.clear_value.to_vk()),
            )
        }

        let mut rendering_info = vk::RenderingInfo::default()
            .flags(rendering_begin_info.rendering_flags.to_vk())
            .color_attachments(color_attachment_info.as_slice())
            .layer_count(rendering_begin_info.layer_count)
            .view_mask(rendering_begin_info.view_mask)
            .render_area(vk::Rect2D {
                extent: rendering_begin_info.render_area.extent.to_vk(),
                offset: rendering_begin_info.render_area.offset.to_vk(),
            });

        let depth_attachment_info: vk::RenderingAttachmentInfo;
        let stencil_attachment_info: vk::RenderingAttachmentInfo;

        // Adding the optinal depth and stencil attachment
        if let Some(depth_attachment) = rendering_begin_info.depth_attachment {
            depth_attachment_info = vk::RenderingAttachmentInfo::default()
                .resolve_mode(depth_attachment.resolve_mode.to_vk())
                .image_view(depth_attachment.image_view.raw)
                .image_layout(match depth_attachment.store_op {
                    StoreOp::None => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    _ => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                })
                .resolve_image_view(depth_attachment.resolve_image_view.map(|i| i.raw).unwrap_or(vk::ImageView::null()))
                .resolve_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(depth_attachment.load_op.to_vk())
                .store_op(depth_attachment.store_op.to_vk())
                .clear_value(depth_attachment.clear_value.to_vk());

            rendering_info = rendering_info.depth_attachment(&depth_attachment_info);
        }

        if let Some(stencil_attachment) = rendering_begin_info.stencil_attachment {
            stencil_attachment_info = vk::RenderingAttachmentInfo::default()
                .resolve_mode(stencil_attachment.resolve_mode.to_vk())
                .image_view(stencil_attachment.image_view.raw)
                .image_layout(match stencil_attachment.store_op {
                    StoreOp::None => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    _ => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                })
                .resolve_image_view(stencil_attachment.resolve_image_view.map(|i| i.raw).unwrap_or(vk::ImageView::null()))
                .resolve_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(stencil_attachment.load_op.to_vk())
                .store_op(stencil_attachment.store_op.to_vk())
                .clear_value(stencil_attachment.clear_value.to_vk());

            rendering_info = rendering_info.stencil_attachment(&stencil_attachment_info);
        }

        unsafe {
            ctx.device.handle.cmd_begin_rendering(self.handle, &rendering_info);
        }

        f(RenderRecorder {
            cmd_buffer: self.handle,
        });

        unsafe {
            ctx.device.handle.cmd_end_rendering(self.handle);
        }
    }
}
