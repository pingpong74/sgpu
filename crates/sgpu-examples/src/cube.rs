use std::time::Duration;

use glam::{Mat4, Vec3, camera::rh};
use sgpu::*;
use sgpu_examples::*;

const SHADERS_SPV: &[u8] = include_bytes!(env!("SHADERS_SPV"));

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    vertex_buffer: u32,
    mvp: [f32; 16],
}

struct App {
    swapchain: Swapchain,
    vertex_buffer: Buffer,
    depth_image: Image,
    pipeline: RasterizationPipeline,
}

fn cube_vertices() -> [Vertex; 36] {
    let h = 0.5;
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.85, 0.2, 0.2],
            [
                [h, -h, -h],
                [h, h, -h],
                [h, -h, h],
                [h, h, h],
            ],
        ),
        (
            [0.2, 0.4, 0.85],
            [
                [-h, -h, h],
                [-h, h, h],
                [-h, -h, -h],
                [-h, h, -h],
            ],
        ),
        (
            [0.2, 0.8, 0.3],
            [
                [-h, h, -h],
                [-h, h, h],
                [h, h, -h],
                [h, h, h],
            ],
        ),
        (
            [0.85, 0.75, 0.2],
            [
                [-h, -h, h],
                [-h, -h, -h],
                [h, -h, h],
                [h, -h, -h],
            ],
        ),
        (
            [0.7, 0.3, 0.85],
            [
                [-h, -h, h],
                [h, -h, h],
                [-h, h, h],
                [h, h, h],
            ],
        ),
        (
            [0.2, 0.75, 0.8],
            [
                [h, -h, -h],
                [-h, -h, -h],
                [h, h, -h],
                [-h, h, -h],
            ],
        ),
    ];

    let mut vertices = [Vertex {
        position: [0.0; 3],
        color: [0.0; 3],
    }; 36];
    let order = [
        0usize, 1, 2, 2, 1, 3,
    ];
    for (face_index, (color, corners)) in faces.iter().enumerate() {
        for (i, corner) in order.iter().enumerate() {
            vertices[face_index * 6 + i] = Vertex {
                position: corners[*corner],
                color: *color,
            };
        }
    }
    vertices
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

impl Application for App {
    fn new(window: &winit::window::Window) -> Self {
        sgpu_init(&SgpuInititizationInfo::default_from_window(window));

        let size = window.inner_size();
        let swapchain = create_swapchain(
            window,
            &SwapchainDescription {
                format: Format::Rgba16Float,
                frames_in_flight: 1,
                width: size.width,
                height: size.height,
            },
        );

        let pipeline = create_rasterization_pipeline(&RasterizationPipelineDescription {
            vertex: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "cube::cube_vs",
            },
            fragment: ShaderModule {
                spirv: SHADERS_SPV,
                entry: "cube::cube_fs",
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

        let vertex_buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE,
            size: (std::mem::size_of::<Vertex>() * 36) as u64,
            memory_type: MemoryType::HostVisible,
        });
        vertex_buffer.as_mut_slice::<Vertex>().copy_from_slice(&cube_vertices());

        let depth_image = create_depth_image(size.width, size.height);

        return App {
            swapchain,
            vertex_buffer,
            depth_image,
            pipeline,
        };
    }

    fn render(&mut self, window: &winit::window::Window, _dt: Duration, time: Duration) {
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let rotation = Mat4::from_rotation_y(time.as_secs_f32() * 0.7) * Mat4::from_rotation_x(time.as_secs_f32() * 0.4);
        let view = rh::view::look_at_mat4(Vec3::new(2.5, 2.0, 3.5), Vec3::ZERO, Vec3::Y);
        let projection = rh::proj::vulkan::perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let mvp = projection * view * rotation;

        let swapchain_img = self.swapchain.acquire_image();

        let mut recorder = record(QueueType::Graphics);

        recorder.wait_for_swapchain_image(&swapchain_img);

        recorder.image_barrier(&ImageBarrier {
            view: swapchain_img.image().default_view(),
            previous_accesses: &[AccessType::None],
            next_accesses: &[AccessType::ColorAttachmentWrite],
            discard_contents: true,
            ..Default::default()
        });

        recorder.image_barrier(&ImageBarrier {
            view: self.depth_image.default_view(),
            previous_accesses: &[AccessType::None],
            next_accesses: &[AccessType::DepthStencilAttachmentWrite],
            discard_contents: true,
            ..Default::default()
        });

        recorder.begin_rendering(
            &RenderingBeginInfo {
                render_area: RenderArea {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: Extent2D {
                        width: size.width,
                        height: size.height,
                    },
                },
                color_attachments: &[
                    RenderingAttachment {
                        image_view: swapchain_img.image().default_view(),
                        clear_value: ClearValue::ColorFloat([0.08, 0.08, 0.1, 1.0]),
                        ..Default::default()
                    },
                ],
                depth_attachment: Some(RenderingAttachment {
                    image_view: self.depth_image.default_view(),
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
                r.push_constants(&PushConstants {
                    vertex_buffer: self.vertex_buffer.descriptor_index(),
                    mvp: mvp.to_cols_array(),
                });
                r.set_viewport(size.width, size.height);
                r.set_scissor(size.width, size.height);
                r.draw(36, 1, 0, 0);
            },
        );

        recorder.image_barrier(&ImageBarrier {
            view: swapchain_img.image().default_view(),
            previous_accesses: &[AccessType::ColorAttachmentWrite],
            next_accesses: &[AccessType::Present],
            discard_contents: false,
            ..Default::default()
        });

        let render_finish = submit(&[recorder]);

        self.swapchain.present(&swapchain_img, render_finish);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.swapchain.resize(width, height);
        self.depth_image = create_depth_image(width, height);
    }
}

fn main() { run::<App>(); }
