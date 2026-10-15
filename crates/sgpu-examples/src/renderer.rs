use std::time::Duration;

use glam::{Mat4, Vec3, camera::rh};
use sgpu::*;
use sgpu_examples::*;
use sgpu_taskgraph::*;

const SHADERS_SPV: &[u8] = include_bytes!(env!("SHADERS_SPV"));

#[repr(C)]
#[derive(Clone, Copy)]
struct ScenePushConstants {
    texture: SampledImageId,
    sampler: SamplerId,
    mvp: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PostPushConstants {
    hdr: SampledImageId,
    sampler: SamplerId,
}

struct Res {
    hdr: Id<Image>,
    depth: Id<Image>,
    swapchain: Id<Image>,
}

struct Frame {
    extent: Extent2D,
    mvp: [f32; 16],
}

struct SceneConfig {
    pipeline: RasterizationPipeline,
    texture: Image,
    sampler: Sampler,
}

struct Scene {
    hdr: PassImage<ColorWrite>,
    depth: PassImage<DepthWrite>,
    pipeline: RasterizationPipeline,
    texture: SampledImageId,
    sampler: SamplerId,
}

impl Node<Res, Frame> for Scene {
    const QUEUE: QueueType = QueueType::Graphics;
    type Config = SceneConfig;

    fn declare(res: &Res, config: &SceneConfig, w: &mut NodeWriter) -> Self {
        Scene {
            hdr: w.image::<ColorWrite>(res.hdr),
            depth: w.image::<DepthWrite>(res.depth),
            pipeline: config.pipeline,
            texture: config.texture.default_view().sampled_id(),
            sampler: config.sampler.descriptor_index(),
        }
    }

    fn execute(&self, data: &Frame, ctx: &mut ExecutionContext<'_>) {
        let hdr = ctx.view(self.hdr);
        let depth = ctx.view(self.depth);

        ctx.cmd().begin_rendering(
            &RenderingBeginInfo {
                render_area: RenderArea {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: data.extent,
                },
                color_attachments: &[
                    RenderingAttachment {
                        image_view: hdr,
                        clear_value: ClearValue::ColorFloat([0.08, 0.08, 0.1, 1.0]),
                        ..Default::default()
                    },
                ],
                depth_attachment: Some(RenderingAttachment {
                    image_view: depth,
                    clear_value: ClearValue::DepthStencil {
                        depth: 1.0,
                        stencil: 0,
                    },
                    ..Default::default()
                }),
                ..Default::default()
            },
            |r| {
                r.bind_rasterization_pipeline(&self.pipeline);
                r.push_constants(&ScenePushConstants {
                    texture: self.texture,
                    sampler: self.sampler,
                    mvp: data.mvp,
                });
                r.set_viewport(data.extent.width, data.extent.height);
                r.set_scissor(data.extent.width, data.extent.height);
                r.draw(36, 1, 0, 0);
            },
        );
    }
}

struct PostConfig {
    pipeline: RasterizationPipeline,
    sampler: Sampler,
}

struct Post {
    swapchain: PassImage<ColorWrite>,
    hdr: PassImage<FragmentSampledRead>,
    pipeline: RasterizationPipeline,
    sampler: SamplerId,
}

impl Node<Res, Frame> for Post {
    const QUEUE: QueueType = QueueType::Graphics;
    type Config = PostConfig;

    fn declare(res: &Res, config: &PostConfig, w: &mut NodeWriter) -> Self {
        Post {
            swapchain: w.image::<ColorWrite>(res.swapchain),
            hdr: w.image::<FragmentSampledRead>(res.hdr),
            pipeline: config.pipeline,
            sampler: config.sampler.descriptor_index(),
        }
    }

    fn execute(&self, data: &Frame, ctx: &mut ExecutionContext<'_>) {
        let swapchain = ctx.view(self.swapchain);
        let hdr = ctx.sampled_id(self.hdr);
        let extent = data.extent;

        ctx.cmd().begin_rendering(
            &RenderingBeginInfo {
                render_area: RenderArea {
                    offset: Offset2D { x: 0, y: 0 },
                    extent,
                },
                color_attachments: &[
                    RenderingAttachment {
                        image_view: swapchain,
                        clear_value: ClearValue::ColorFloat([0.0, 0.0, 0.0, 1.0]),
                        ..Default::default()
                    },
                ],
                depth_attachment: None,
                ..Default::default()
            },
            |r| {
                r.bind_rasterization_pipeline(&self.pipeline);
                r.push_constants(&PostPushConstants {
                    hdr,
                    sampler: self.sampler,
                });
                r.set_viewport(extent.width, extent.height);
                r.set_scissor(extent.width, extent.height);
                r.draw(3, 1, 0, 0);
            },
        );
    }
}

struct PresentPass {
    swapchain: PassImage<Present>,
}

impl Node<Res, Frame> for PresentPass {
    const QUEUE: QueueType = QueueType::Graphics;
    type Config = ();

    fn declare(res: &Res, _: &(), w: &mut NodeWriter) -> Self {
        PresentPass {
            swapchain: w.image::<Present>(res.swapchain),
        }
    }

    fn execute(&self, _: &Frame, _: &mut ExecutionContext<'_>) { let _swapchain = self.swapchain; }
}

fn create_hdr_image(width: u32, height: u32) -> Image {
    create_image(&ImageDescription {
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
        format: Format::Rgba16Float,
        extent: Extent3D {
            width,
            height,
            depth: 1,
        },
        memory_type: MemoryType::DeviceLocal,
        ..Default::default()
    })
}

fn create_depth_image(width: u32, height: u32) -> Image {
    create_image(&ImageDescription {
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        format: Format::D32Float,
        extent: Extent3D {
            width,
            height,
            depth: 1,
        },
        memory_type: MemoryType::DeviceLocal,
        ..Default::default()
    })
}

const MOSS_PNG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/res/moss_block.png"));

fn create_moss_texture() -> Image {
    let decoded = image::load_from_memory(MOSS_PNG).expect("Failed to decode moss_block.png").to_rgba8();
    let (width, height) = decoded.dimensions();

    let staging = create_buffer(&BufferDescription {
        usage: BufferUsage::TRANSFER_SRC,
        size: decoded.len() as u64,
        memory_type: MemoryType::HostVisible,
    });
    buffer_as_slice_mut::<u8>(&staging).copy_from_slice(&decoded);

    let texture = create_image(&ImageDescription {
        usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
        format: Format::Rgba8Unorm,
        extent: Extent3D {
            width,
            height,
            depth: 1,
        },
        memory_type: MemoryType::DeviceLocal,
        ..Default::default()
    });

    let mut recorder = record(QueueType::Graphics);
    recorder.image_barrier(&ImageBarrier {
        view: texture.default_view(),
        previous_accesses: &[AccessType::None],
        next_accesses: &[AccessType::TransferWrite],
        discard_contents: true,
        ..Default::default()
    });
    recorder.copy_buffer_to_image(
        &staging,
        &texture,
        &[
            BufferImageCopyRegion {
                image_extent: Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                ..Default::default()
            },
        ],
    );
    recorder.image_barrier(&ImageBarrier {
        view: texture.default_view(),
        previous_accesses: &[AccessType::TransferWrite],
        next_accesses: &[AccessType::FragmentShaderSampledRead],
        discard_contents: false,
        ..Default::default()
    });

    wait(submit(&[recorder]));

    destroy_buffer(staging);
    texture
}

struct App {
    swapchain: Swapchain,
    graph: Graph<'static, Res, Frame>,
    res: Res,
    hdr_image: Image,
    depth_image: Image,
    texture: Image,
    sampler: Sampler,
    scene_pipeline: RasterizationPipeline,
    post_pipeline: RasterizationPipeline,
}

impl Application for App {
    fn new(window: &winit::window::Window) -> Self {
        sgpu_init(&SgpuInititizationInfo::default_from_window(window));

        let size = window.inner_size();
        let swapchain = create_swapchain(
            window,
            &SwapchainDescription {
                format: Format::Rgba16Float,
                frames_in_flight: 3,
                width: size.width,
                height: size.height,
            },
        );

        let scene_pipeline = create_rasterization_pipeline(&RasterizationPipelineDescription {
            vertex: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "cube::cube_vs",
                ..Default::default()
            },
            fragment: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "cube::cube_fs",
                ..Default::default()
            },
            topology: PrimitiveTopology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            polygon_mode: PolygonMode::Fill,
            depth_stencil: DepthStencilState::default(),
            blend_mode: BlendMode::Opaque,
            outputs: PipelineOutputs {
                color: &[Format::Rgba16Float],
                depth: Some(Format::D32Float),
                stencil: None,
            },
        });

        let post_pipeline = create_rasterization_pipeline(&RasterizationPipelineDescription {
            vertex: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "post::post_vs",
                ..Default::default()
            },
            fragment: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "post::post_fs",
                ..Default::default()
            },
            topology: PrimitiveTopology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            polygon_mode: PolygonMode::Fill,
            depth_stencil: DepthStencilState::DISABLED,
            blend_mode: BlendMode::Opaque,
            outputs: PipelineOutputs {
                color: &[Format::Rgba16Float],
                depth: None,
                stencil: None,
            },
        });

        let hdr_image = create_hdr_image(size.width, size.height);
        let depth_image = create_depth_image(size.width, size.height);
        let texture = create_moss_texture();
        let sampler = create_sampler(&SamplerDescription {
            filter: FilterMode::Nearest,
            ..Default::default()
        });

        let mut graph = Graph::new();
        let res = Res {
            swapchain: graph.add_image(Image::default()),
            hdr: graph.add_image(hdr_image),
            depth: graph.add_image(depth_image),
        };
        graph.add_node::<Scene>(SceneConfig {
            pipeline: scene_pipeline,
            texture,
            sampler,
        });
        graph.add_node::<Post>(PostConfig {
            pipeline: post_pipeline,
            sampler,
        });
        graph.add_node::<PresentPass>(());

        App {
            swapchain,
            graph,
            res,
            hdr_image,
            depth_image,
            texture,
            sampler,
            scene_pipeline,
            post_pipeline,
        }
    }

    fn render(&mut self, window: &winit::window::Window, _dt: Duration, time: Duration) {
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let rotation = Mat4::from_rotation_y(time.as_secs_f32() * 0.7) * Mat4::from_rotation_x(time.as_secs_f32() * 0.4);
        let view = rh::view::look_at_mat4(Vec3::new(2.5, 2.0, 3.5), Vec3::ZERO, Vec3::Y);
        let projection = rh::proj::vulkan::perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let mvp = projection * view * rotation;

        let acquired = self.swapchain.acquire_image();
        self.graph.set_image(self.res.swapchain, acquired.image());

        let counter = self.graph.execute_with_swapchain(
            &self.res,
            &Frame {
                extent: Extent2D {
                    width: size.width,
                    height: size.height,
                },
                mvp: mvp.to_cols_array(),
            },
            &acquired,
        );

        self.swapchain.present(&acquired, counter);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.swapchain.resize(width, height);

        destroy_image(self.hdr_image);
        self.hdr_image = create_hdr_image(width, height);
        destroy_image(self.depth_image);
        self.depth_image = create_depth_image(width, height);
        self.graph.set_image(self.res.hdr, self.hdr_image);
        self.graph.set_image(self.res.depth, self.depth_image);

        self.graph.reset_steady_state();
    }
}

impl Drop for App {
    fn drop(&mut self) {
        wait_idle();
        destroy_image(self.hdr_image);
        destroy_image(self.depth_image);
        destroy_image(self.texture);
        destroy_sampler(self.sampler);
        destroy_rasterization_pipeline(self.scene_pipeline);
        destroy_rasterization_pipeline(self.post_pipeline);
    }
}

fn main() { run::<App>(); }
