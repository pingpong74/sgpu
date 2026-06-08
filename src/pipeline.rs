use ash::vk;

#[derive(Clone, Copy, Debug, Default)]
pub enum CompareOp {
    Never,
    #[default]
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}
impl CompareOp {
    pub(crate) fn to_vk(self) -> vk::CompareOp {
        match self {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
    FrontAndBack,
}

impl CullMode {
    pub(crate) const fn to_vk(self) -> vk::CullModeFlags {
        match self {
            Self::None => vk::CullModeFlags::NONE,
            Self::Front => vk::CullModeFlags::FRONT,
            Self::Back => vk::CullModeFlags::BACK,
            Self::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

impl FrontFace {
    pub(crate) const fn to_vk(self) -> vk::FrontFace {
        match self {
            Self::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            Self::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

impl PolygonMode {
    pub(crate) fn to_vk(self) -> vk::PolygonMode {
        match self {
            Self::Fill => vk::PolygonMode::FILL,
            Self::Line => vk::PolygonMode::LINE,
            Self::Point => vk::PolygonMode::POINT,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

impl PrimitiveTopology {
    pub(crate) const fn to_vk(self) -> vk::PrimitiveTopology {
        match self {
            Self::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::LineList => vk::PrimitiveTopology::LINE_LIST,
            Self::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Self::PointList => vk::PrimitiveTopology::POINT_LIST,
        }
    }
}

#[derive(Clone, Copy)]
pub struct DepthStencilState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: CompareOp,
    pub stencil_test: bool,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test: true,
            depth_write: true,
            depth_compare: CompareOp::Less,
            stencil_test: false,
        }
    }
}

impl DepthStencilState {
    pub const DISABLED: Self = Self {
        depth_test: false,
        depth_write: false,
        depth_compare: CompareOp::Always,
        stencil_test: false,
    };

    pub const READ_ONLY: Self = Self {
        depth_test: true,
        depth_write: false,
        depth_compare: CompareOp::Less,
        stencil_test: false,
    };
}

#[derive(Clone, Copy, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    Alpha,
    Additive,
    Premultiplied,
}

impl BlendMode {
    pub(crate) fn to_vk_attachment(self) -> vk::PipelineColorBlendAttachmentState {
        let full_write = vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A;

        match self {
            Self::Opaque => vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: full_write,
                ..Default::default()
            },
            Self::Alpha => vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: full_write,
            },
            Self::Additive => vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: full_write,
            },
            Self::Premultiplied => vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: full_write,
            },
        }
    }
}

/// The render targets this pipeline will write to.
/// Must match the attachments in RenderingBeginInfo at draw time.
#[derive(Clone)]
pub struct PipelineOutputs<'a> {
    pub color: &'a [crate::Format],
    pub depth: Option<crate::Format>,
    pub stencil: Option<crate::Format>,
}

impl<'a> Default for PipelineOutputs<'a> {
    fn default() -> Self {
        Self {
            color: &[crate::Format::Rgba16Float],
            depth: None,
            stencil: None,
        }
    }
}

pub struct RasterizationPipelineDescription<'a> {
    pub vertex_shader: &'a [u8],
    pub fragment_shader: &'a [u8],

    pub topology: PrimitiveTopology,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub polygon_mode: PolygonMode,
    pub depth_stencil: DepthStencilState,
    pub blend_mode: BlendMode,
    pub outputs: PipelineOutputs<'a>,
}

impl<'a> Default for RasterizationPipelineDescription<'a> {
    fn default() -> Self {
        Self {
            vertex_shader: &[],
            fragment_shader: &[],
            topology: PrimitiveTopology::default(),
            cull_mode: CullMode::default(),
            front_face: FrontFace::default(),
            polygon_mode: PolygonMode::default(),
            depth_stencil: DepthStencilState::default(),
            blend_mode: BlendMode::default(),
            outputs: PipelineOutputs::default(),
        }
    }
}

pub struct RasterizationPipeline {
    pub(crate) handle: vk::Pipeline,
}

impl Drop for RasterizationPipeline {
    fn drop(&mut self) {
        unsafe {
            crate::CONTEXT.get().unwrap().device.handle.destroy_pipeline(self.handle, None);
        }
    }
}

pub struct ComputePipeline {}
