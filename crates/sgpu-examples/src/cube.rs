use std::time::Duration;

use glam::{Mat4, Vec3, camera::rh};
use sgpu::*;
use sgpu_examples::*;

const SHADERS_SPV: &[u8] = include_bytes!(env!("SHADERS_SPV"));

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    texture: SampledImageId,
    sampler: SamplerId,
    mvp: [f32; 16],
}

struct App {
    swapchain: Swapchain,
    depth_image: Image,
    texture: Image,
    sampler: Sampler,
    pipeline: RasterizationPipeline,
    timestamp_pool: QueryPool,
    frame: u64,
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

        let pipeline = create_rasterization_pipeline(&RasterizationPipelineDescription {
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

        let depth_image = create_depth_image(size.width, size.height);
        let texture = create_moss_texture();
        let sampler = create_sampler(&SamplerDescription {
            filter: FilterMode::Nearest,
            ..Default::default()
        });
        let timestamp_pool = create_query_pool(2);

        App {
            swapchain,
            depth_image,
            texture,
            sampler,
            pipeline,
            timestamp_pool,
            frame: 0,
        }
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

        recorder.reset_query_pool(&self.timestamp_pool, 0, 2);
        recorder.write_timestamp(&self.timestamp_pool, 0, PipelineStage::TOP_OF_PIPE);

        recorder.wait_for_swapchain_image(&swapchain_img);

        recorder.barriers(
            None,
            &[
                ImageBarrier {
                    view: swapchain_img.image().default_view(),
                    previous_accesses: &[AccessType::None],
                    next_accesses: &[AccessType::ColorAttachmentWrite],
                    discard_contents: true,
                    ..Default::default()
                },
                ImageBarrier {
                    view: self.depth_image.default_view(),
                    previous_accesses: &[AccessType::None],
                    next_accesses: &[AccessType::DepthStencilAttachmentWrite],
                    discard_contents: true,
                    ..Default::default()
                },
            ],
        );

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
                    texture: self.texture.default_view().sampled_id(),
                    sampler: self.sampler.descriptor_index(),
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

        recorder.write_timestamp(&self.timestamp_pool, 1, PipelineStage::BOTTOM_OF_PIPE);

        let render_finish = submit(&[recorder]);

        self.swapchain.present(&swapchain_img, render_finish);

        self.frame += 1;
        if self.frame.is_multiple_of(60) {
            wait(render_finish);
            let timestamps = self.timestamp_pool.get_timestamps_ns(0, 2);
            print!("\rGPU frame: {:.3} ms", (timestamps[1] - timestamps[0]) / 1_000_000.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.swapchain.resize(width, height);
        destroy_image(self.depth_image);
        self.depth_image = create_depth_image(width, height);
    }
}

impl Drop for App {
    fn drop(&mut self) {
        wait_idle();
        destroy_image(self.depth_image);
        destroy_image(self.texture);
        destroy_sampler(self.sampler);
        destroy_rasterization_pipeline(self.pipeline);
    }
}

fn main() { run::<App>(); }
