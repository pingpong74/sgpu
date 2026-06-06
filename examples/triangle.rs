mod common;

use common::*;
use raw_window_handle::HasWindowHandle;
use sgpu::*;

struct App {
    swapchain: Swapchain,
}

impl Application for App {
    fn new(window: &winit::window::Window) -> Self {
        sgpu_init(&SgpuInititizationInfo::default_from_window(window));

        let size = window.inner_size();
        let swapchain = create_swapchain(
            window,
            &SwapchainDescription {
                format: Format::Rgba16Float,
                frames_in_flight: 2,
                width: size.width,
                height: size.height,
            },
        );

        return App {
            swapchain: swapchain,
        };
    }

    fn render(&mut self, window: &winit::window::Window) {
        let size = window.inner_size();
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
            |_| {},
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
    run::<App>();
}
