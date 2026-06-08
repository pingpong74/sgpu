use crate::{
    backend::{Context, InnerSwapchain, Surface},
    commands::{CommandBuffer, QueueType},
    pipeline::{RasterizationPipeline, RasterizationPipelineDescription},
    swapchain::{Swapchain, SwapchainDescription},
    types::*,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct SgpuInititizationInfo {
    pub app_name: &'static str,
    pub enable_validation_layers: bool,
    pub window_handle: Option<raw_window_handle::RawWindowHandle>,

    pub mesh_shaders: bool,
    pub atomic_float_operations: bool,
    pub ray_tracing: bool,
}

impl SgpuInititizationInfo {
    pub fn default_from_window<W: HasWindowHandle>(window: &W) -> SgpuInititizationInfo {
        return SgpuInititizationInfo {
            window_handle: Some(window.window_handle().unwrap().as_raw()),
            ..Default::default()
        };
    }
}

impl Default for SgpuInititizationInfo {
    fn default() -> Self {
        return Self {
            app_name: "Default",
            enable_validation_layers: true,
            window_handle: None,
            mesh_shaders: false,
            atomic_float_operations: false,
            ray_tracing: false,
        };
    }
}

#[derive(Debug)]
pub enum SgpuInitError {}

/// Main function, needs to be called before everythign else
pub fn sgpu_init(init_info: &SgpuInititizationInfo) {
    let _ = crate::CONTEXT.set(Context::new(init_info));
}

pub fn create_buffer(buffer_desc: &BufferDescription) -> Buffer {
    return crate::CONTEXT.get().expect("Not initialized").create_buffer(buffer_desc);
}

pub fn destroy_buffer(buffer: Buffer) {
    crate::CONTEXT.get().expect("Not initialized").destroy_buffer(buffer);
}

pub fn create_image(image_desc: &ImageDescription) -> Image {
    return crate::CONTEXT.get().expect("Not initialized").create_image(image_desc);
}

pub fn destroy_image(image: Image) {
    crate::CONTEXT.get().expect("Not initialized").destroy_image(image);
}

pub fn create_swapchain<W: HasDisplayHandle + HasWindowHandle>(window: &W, swapchain_description: &SwapchainDescription) -> Swapchain {
    let surface = Surface::create_surface(&crate::CONTEXT.get().expect("Not initialized").instance, window);
    let inner = InnerSwapchain::new(&surface, swapchain_description, None);

    return Swapchain {
        inner: inner,
        surface: surface,
    };
}

pub fn create_rasterization_pipeline(raster_pipeline_desc: &RasterizationPipelineDescription) -> RasterizationPipeline {
    return crate::CONTEXT.get().unwrap().create_raster_pipeline(raster_pipeline_desc);
}

/// A simple function to check if the counter has already been signaled
/// Not waits or anything
pub fn poll(counter: Counter) -> bool {
    return crate::CONTEXT.get().expect("Not initialized").poll(counter);
}

/// Wait for a counter to be signaled CPU side.
pub fn wait(counter: Counter) {
    crate::CONTEXT.get().expect("Not initialized").wait(counter);
}

/// wait for all gpu work to be over
pub fn wait_idle() {
    crate::CONTEXT.get().expect("Not initialized").wait_idle();
}

/// began the recording of command buffer
pub fn record(queue_type: QueueType) -> CommandBuffer {
    return crate::CONTEXT.get().expect("Not initialized").record(queue_type);
}

/// submit command buffers
pub fn submit(command_buffers: &[CommandBuffer]) -> Counter {
    return crate::CONTEXT.get().expect("Not initialized").submit(command_buffers);
}
