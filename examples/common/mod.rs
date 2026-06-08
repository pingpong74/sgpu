use std::time::{Duration, Instant};
use winit::{application::*, event::*, event_loop::*, window::*};

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
        return Runner {
            app: None,
            window: None,
            start_time: Instant::now(),
            last_frame: Instant::now(),
        };
    }
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
