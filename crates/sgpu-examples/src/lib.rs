use sgpu::{AccessType, BufferDescription, BufferImageCopyRegion, BufferUsage, Extent3D, Format, Image, ImageBarrier, ImageDescription, ImageUsage, MemoryType, QueueType};
use std::time::{Duration, Instant};
use winit::{application::*, event::*, event_loop::*, window::*};

pub fn create_depth_image(width: u32, height: u32) -> Image {
    sgpu::create_image(&ImageDescription {
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

pub fn create_moss_texture() -> Image {
    let decoded = image::load_from_memory(MOSS_PNG).expect("Failed to decode moss_block.png").to_rgba8();
    let (width, height) = decoded.dimensions();

    let staging = sgpu::create_buffer(&BufferDescription {
        usage: BufferUsage::TRANSFER_SRC,
        size: decoded.len() as u64,
        memory_type: MemoryType::HostVisible,
    });
    sgpu::buffer_as_slice_mut::<u8>(&staging).copy_from_slice(&decoded);

    let texture = sgpu::create_image(&ImageDescription {
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

    let mut recorder = sgpu::record(QueueType::Graphics);
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

    sgpu::wait(sgpu::submit(&[recorder]));

    sgpu::destroy_buffer(staging);
    texture
}

pub trait Application {
    fn new(window: &Window) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn render(&mut self, window: &Window, delta_time: Duration, elapsed_time: Duration);
}

pub struct Runner<A: Application> {
    app: Option<A>,
    window: Option<Window>,

    start_time: Instant,
    last_frame: Instant,
}

impl<A: Application> Runner<A> {
    pub fn new() -> Runner<A> {
        Runner {
            app: None,
            window: None,
            start_time: Instant::now(),
            last_frame: Instant::now(),
        }
    }
}

impl<A: Application> Default for Runner<A> {
    fn default() -> Self { Self::new() }
}

impl<A: Application> ApplicationHandler for Runner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }

        let window = event_loop.create_window(WindowAttributes::default()).expect("Failed to create window");
        window.set_cursor_grab(winit::window::CursorGrabMode::Locked).expect(":(");
        window.set_cursor_visible(false);

        self.start_time = Instant::now();
        self.last_frame = Instant::now();

        self.app = Some(Application::new(&window));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if self.window.is_none() || self.app.is_none() {
            return;
        }

        let window = self.window.as_ref().unwrap();
        let app = self.app.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();

                let delta_time = now - self.last_frame;
                let elapsed_time = now - self.start_time;

                self.last_frame = now;

                app.render(window, delta_time, elapsed_time);

                window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                app.resize(size.width, size.height);
            }
            _ => {}
        }
    }
}

pub fn run<A: Application>() {
    let event_loop = EventLoop::new().unwrap();

    let mut runner = Runner::<A>::new();

    event_loop.run_app(&mut runner).expect("Meow");
}
