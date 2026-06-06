use ash::vk;
use smallvec::SmallVec;

use crate::{ImageView, commands::CommandBuffer};

/// AccessType, maps to vk::AccessFlags2
/// Used an enum so Image Layout's can be inffered
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum AccessType {
    #[default]
    None,

    IndirectBuffer,
    IndexBuffer,
    VertexBuffer,
    VertexShaderUniformRead,
    VertexShaderSampledRead,
    VertexShaderStorageRead,

    // Tessellation control
    TessellationControlShaderUniformRead,
    TessellationControlShaderSampledRead,
    TessellationControlShaderStorageRead,

    // Tessellation evaluation
    TessellationEvaluationShaderUniformRead,
    TessellationEvaluationShaderSampledRead,
    TessellationEvaluationShaderStorageRead,

    // Geometry
    GeometryShaderUniformRead,
    GeometryShaderSampledRead,
    GeometryShaderStorageRead,

    // Fragment
    FragmentShaderUniformRead,
    FragmentShaderSampledRead,
    FragmentShaderColorInputAttachmentRead,
    FragmentShaderDepthStencilInputAttachmentRead,
    FragmentShaderStorageRead,

    // Fixed-function attachment reads
    ColorAttachmentRead,
    DepthStencilAttachmentRead,

    // Compute
    ComputeShaderUniformRead,
    ComputeShaderSampledRead,
    ComputeShaderStorageRead,

    // Any shader
    AnyShaderUniformRead,
    AnyShaderSampledRead,
    AnyShaderStorageRead,

    // Transfer / host
    TransferRead,
    HostRead,

    // Present
    Present,

    // writes
    VertexShaderStorageWrite,
    TessellationControlShaderStorageWrite,
    TessellationEvaluationShaderStorageWrite,
    GeometryShaderStorageWrite,
    FragmentShaderStorageWrite,
    ColorAttachmentWrite,
    ColorAttachmentReadWrite,
    DepthStencilAttachmentWrite,
    DepthAttachmentWriteStencilReadOnly,
    StencilAttachmentWriteDepthReadOnly,
    ComputeShaderStorageWrite,
    AnyShaderStorageWrite,
    TransferWrite,
    HostWrite,
    RayTracingShaderSampledRead,
    RayTracingShaderStorageRead,
    RayTracingShaderColorInputAttachmentRead,
    RayTracingShaderDepthStencilInputAttachmentRead,
    RayTracingShaderAccelerationStructureRead,
    AccelerationStructureBuildWrite,
    AccelerationStructureBuildRead,
    AccelerationStructureBufferWrite,
    General,
}

pub(crate) struct AccessInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub image_layout: vk::ImageLayout,
}

impl AccessType {
    pub(crate) fn info(self) -> AccessInfo {
        match self {
            Self::None => AccessInfo {
                stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                access: vk::AccessFlags2::NONE,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::IndirectBuffer => AccessInfo {
                stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::IndexBuffer => AccessInfo {
                stage: vk::PipelineStageFlags2::INDEX_INPUT,
                access: vk::AccessFlags2::INDEX_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::VertexBuffer => AccessInfo {
                stage: vk::PipelineStageFlags2::VERTEX_INPUT,
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::VertexShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::VertexShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::VertexShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TessellationControlShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::TessellationControlShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::TessellationControlShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TessellationEvaluationShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::TessellationEvaluationShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::TessellationEvaluationShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::GeometryShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::GeometryShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::GeometryShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::FragmentShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::FragmentShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::FragmentShaderColorInputAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::FragmentShaderDepthStencilInputAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            Self::FragmentShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::ColorAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            Self::DepthStencilAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            Self::ComputeShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::ComputeShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::ComputeShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::AnyShaderUniformRead => AccessInfo {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::AnyShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::AnyShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TransferRead => AccessInfo {
                stage: vk::PipelineStageFlags2::COPY,
                access: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            Self::HostRead => AccessInfo {
                stage: vk::PipelineStageFlags2::HOST,
                access: vk::AccessFlags2::HOST_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::Present => AccessInfo {
                stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                access: vk::AccessFlags2::NONE,
                image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            Self::VertexShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TessellationControlShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TessellationEvaluationShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::GeometryShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::FragmentShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::ColorAttachmentWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            Self::ColorAttachmentReadWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            Self::DepthStencilAttachmentWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            Self::DepthAttachmentWriteStencilReadOnly => AccessInfo {
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            },
            Self::StencilAttachmentWriteDepthReadOnly => AccessInfo {
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
            },
            Self::ComputeShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::AnyShaderStorageWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::TransferWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::COPY,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            Self::HostWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::HOST,
                access: vk::AccessFlags2::HOST_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::RayTracingShaderSampledRead => AccessInfo {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::RayTracingShaderStorageRead => AccessInfo {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            Self::RayTracingShaderColorInputAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Self::RayTracingShaderDepthStencilInputAttachmentRead => AccessInfo {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            Self::RayTracingShaderAccelerationStructureRead => AccessInfo {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::AccelerationStructureBuildWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::AccelerationStructureBuildRead => AccessInfo {
                stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::AccelerationStructureBufferWrite => AccessInfo {
                stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            Self::General => AccessInfo {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
        }
    }

    pub fn is_write(self) -> bool {
        matches!(
            self,
            Self::VertexShaderStorageWrite
                | Self::TessellationControlShaderStorageWrite
                | Self::TessellationEvaluationShaderStorageWrite
                | Self::GeometryShaderStorageWrite
                | Self::FragmentShaderStorageWrite
                | Self::ColorAttachmentWrite
                | Self::ColorAttachmentReadWrite
                | Self::DepthStencilAttachmentWrite
                | Self::DepthAttachmentWriteStencilReadOnly
                | Self::StencilAttachmentWriteDepthReadOnly
                | Self::ComputeShaderStorageWrite
                | Self::AnyShaderStorageWrite
                | Self::TransferWrite
                | Self::HostWrite
                | Self::AccelerationStructureBuildWrite
                | Self::AccelerationStructureBufferWrite
                | Self::General
        )
    }
}

/// Simplified VkImageLayout
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum ImageLayout {
    /// Choose the most optimal layout for each usage. Performs layout transitions as appropriate for the access.
    #[default]
    Optimal,

    /// Layout accessible by all Vulkan access types on a device - no layout transitions except for presentation
    General,

    /// Similar to `General`, but also allows presentation engines to access it
    GeneralAndPresentation,
}

impl ImageLayout {
    fn resolve(self, info: &AccessInfo, is_present: bool) -> vk::ImageLayout {
        match self {
            Self::Optimal => info.image_layout,
            Self::General => {
                if is_present {
                    vk::ImageLayout::PRESENT_SRC_KHR
                } else {
                    vk::ImageLayout::GENERAL
                }
            }
            Self::GeneralAndPresentation => vk::ImageLayout::SHARED_PRESENT_KHR,
        }
    }
}

/// Global memory barrier — no specific resource.
/// Prefer this over image barriers when no layout transition is needed
pub struct GlobalBarrier<'a> {
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
}

/// Image barrier, Only for layout transitions
#[derive(Default)]
pub struct ImageBarrier<'a> {
    pub view: ImageView,
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
    /// Override the layout derived from AccessType. Defaults to Optimal.
    pub previous_layout: ImageLayout,
    /// Override the layout derived from AccessType. Defaults to Optimal.
    pub next_layout: ImageLayout,
    /// Discard previous contents, driver skips the layout blit.
    /// Use on first use or when every texel will be overwritten.
    pub discard_contents: bool,
}

impl CommandBuffer {
    /// Global memory barrier, Use for buffers pr images when no layout transition is needed
    pub fn global_barrier(&mut self, barrier: &GlobalBarrier) {
        let ctx = crate::CONTEXT.get().unwrap();

        let (src_stage, dst_stage, src_access, dst_access) = fold_accesses(barrier.previous_accesses, barrier.next_accesses);

        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);

        unsafe {
            ctx.device
                .handle
                .cmd_pipeline_barrier2(self.handle, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb)));
        }
    }

    /// Image barrier — use when a layout transition is required.
    pub fn image_barrier(&mut self, barrier: &ImageBarrier) {
        let ctx = crate::CONTEXT.get().unwrap();
        let vk_barrier = build_image_barrier(barrier);

        unsafe {
            ctx.device
                .handle
                .cmd_pipeline_barrier2(self.handle, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&vk_barrier)));
        }
    }

    /// Batch multiple barriers into a single cmd_pipeline_barrier2 call.
    /// Always prefer this over calling global_barrier / image_barrier in a loop.
    pub fn barriers(&mut self, global: Option<&GlobalBarrier>, images: &[ImageBarrier]) {
        let img_barriers: SmallVec<[vk::ImageMemoryBarrier2; 8]> = images.iter().map(build_image_barrier).collect();

        let memory_barrier = global.map(|g| {
            let (src_stage, dst_stage, src_access, dst_access) = fold_accesses(g.previous_accesses, g.next_accesses);
            vk::MemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
        });

        let mut dep_info = vk::DependencyInfo::default().image_memory_barriers(&img_barriers);

        if let Some(ref mb) = memory_barrier {
            dep_info = dep_info.memory_barriers(std::slice::from_ref(mb));
        }

        let ctx = crate::CONTEXT.get().unwrap();
        unsafe {
            ctx.device.handle.cmd_pipeline_barrier2(self.handle, &dep_info);
        }
    }
}

/// Fold previous/next access slices into the four Sync2 fields.
/// Implements the WAR optimisation from vk-sync-rs: if src_access is empty
/// (pure execution dependency) dst_access is zeroed too — no visibility op needed.
fn fold_accesses(previous: &[AccessType], next: &[AccessType]) -> (vk::PipelineStageFlags2, vk::PipelineStageFlags2, vk::AccessFlags2, vk::AccessFlags2) {
    let mut src_stage = vk::PipelineStageFlags2::empty();
    let mut dst_stage = vk::PipelineStageFlags2::empty();
    let mut src_access = vk::AccessFlags2::empty();
    let mut dst_access = vk::AccessFlags2::empty();

    for &prev in previous {
        let info = prev.info();
        src_stage |= info.stage;
        if prev.is_write() {
            src_access |= info.access;
        }
    }

    for &next in next {
        let info = next.info();
        dst_stage |= info.stage;
        // WAR — no availability op means no visibility op needed either
        if !src_access.is_empty() {
            dst_access |= info.access;
        }
    }

    if src_stage.is_empty() {
        src_stage = vk::PipelineStageFlags2::TOP_OF_PIPE;
    }
    if dst_stage.is_empty() {
        dst_stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    }

    (src_stage, dst_stage, src_access, dst_access)
}

fn build_image_barrier(barrier: &ImageBarrier) -> vk::ImageMemoryBarrier2<'static> {
    let ctx = crate::CONTEXT.get().unwrap();

    let images = ctx.images.read().unwrap();
    let inner_img = images.get(barrier.view.image_key).unwrap();
    let inner_view = &inner_img.image_views[barrier.view.id];

    let image = inner_img.image;
    let subresource_range = inner_view.subresources;
    drop(images);

    let mut src_stage = vk::PipelineStageFlags2::empty();
    let mut dst_stage = vk::PipelineStageFlags2::empty();
    let mut src_access = vk::AccessFlags2::empty();
    let mut dst_access = vk::AccessFlags2::empty();
    let mut old_layout = vk::ImageLayout::UNDEFINED;
    let mut new_layout = vk::ImageLayout::UNDEFINED;

    for &prev in barrier.previous_accesses {
        let info = prev.info();
        src_stage |= info.stage;
        if prev.is_write() {
            src_access |= info.access;
        }
        if !barrier.discard_contents {
            old_layout = barrier.previous_layout.resolve(&info, prev == AccessType::Present);
        }
    }

    for &next in barrier.next_accesses {
        let info = next.info();
        dst_stage |= info.stage;
        if !src_access.is_empty() {
            dst_access |= info.access;
        }
        new_layout = barrier.next_layout.resolve(&info, next == AccessType::Present);
    }

    if src_stage.is_empty() {
        src_stage = vk::PipelineStageFlags2::TOP_OF_PIPE;
    }
    if dst_stage.is_empty() {
        dst_stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    }

    return vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range);
}
