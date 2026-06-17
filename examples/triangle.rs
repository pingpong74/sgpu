mod common;

use std::time::{Duration, Instant};

use common::*;
use sgpu::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    _pad0: [f32; 2],

    color: [f32; 3],
    _pad1: f32,
}

struct App {
    swapchain: Swapchain,
    veretx_buffer: Buffer,
    pipeline: RasterizationPipeline,
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

        let raster_pipe = create_rasterization_pipeline(&RasterizationPipelineDescription {
            vertex_shader: include_bytes!("shaders/triangle/vertex.spv"),
            fragment_shader: include_bytes!("shaders/triangle/fragment.spv"),
            topology: PrimitiveTopology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::Clockwise,
            polygon_mode: PolygonMode::Fill,
            depth_stencil: DepthStencilState::DISABLED,
            blend_mode: BlendMode::Opaque,
            outputs: PipelineOutputs {
                color: &[Format::Rgba16Float],
                depth: None,
                stencil: None,
            },
        });

        let buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE,
            size: std::mem::size_of::<Vertex>() as u64 * 3,
            memory_type: MemoryType::HostVisible,
        });

        let vertices = buffer.as_mut_slice::<Vertex>();

        // fill all
        vertices[0] = Vertex {
            position: [0.0, -0.5],
            _pad0: [0.0; 2],
            color: [1.0, 0.0, 0.0],
            _pad1: 0.0,
        };

        vertices[1] = Vertex {
            position: [0.5, 0.5],
            _pad0: [0.0; 2],
            color: [0.0, 1.0, 0.0],
            _pad1: 0.0,
        };

        vertices[2] = Vertex {
            position: [-0.5, 0.5],
            _pad0: [0.0; 2],
            color: [0.0, 0.0, 1.0],
            _pad1: 0.0,
        };

        return App {
            swapchain: swapchain,
            veretx_buffer: buffer,
            pipeline: raster_pipe,
        };
    }

    fn render(&mut self, window: &winit::window::Window, dt: Duration, time: Duration) {
        let size = window.inner_size();

        {
            let vertices = self.veretx_buffer.as_mut_slice::<Vertex>();
            let time = time.as_secs_f32();

            let r = time.sin() * 0.5 + 0.5;
            let g = (time + 2.094).sin() * 0.5 + 0.5;
            let b = (time + 4.188).sin() * 0.5 + 0.5;

            vertices[0].color = [r, 0.0, 0.0];
            vertices[1].color = [0.0, g, 0.0];
            vertices[2].color = [0.0, 0.0, b];
        }

        let swapchain_img = self.swapchain.acquire_image();

        println!("{}", 1.0 / dt.as_secs_f32());

        let mut recorder = record(QueueType::Graphics);

        recorder.wait_for_swapchain_image(&swapchain_img);

        recorder.image_barrier(&ImageBarrier {
            view: swapchain_img.image().default_view(),
            previous_accesses: &[AccessType::None],
            next_accesses: &[AccessType::ColorAttachmentWrite],
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
                        clear_value: ClearValue::ColorFloat([0.5, 0.8, 0.4, 1.0]),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            |r| {
                r.bind_rasterization_pipeline(&self.pipeline);
                r.push_constants(&self.veretx_buffer.descriptor_index());
                r.set_viewport(size.width, size.height);
                r.set_scissor(size.width, size.height);
                r.draw(3, 1, 0, 0);
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
    }
}

fn main() {
    add_shader_directory("examples/shaders/triangle/");
    run::<App>();
}
