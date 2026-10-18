use crate::{
    ComputePipeline,
    backend::{Context, InnerSwapchain, Surface},
    commands::{CommandBuffer, QueueType},
    pipeline::{RasterizationPipeline, RasterizationPipelineDescription, ShaderModule},
    query::QueryPool,
    swapchain::{Swapchain, SwapchainDescription},
    types::*,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct SgpuInititizationInfo {
    pub app_name: &'static str,
    pub enable_validation_layers: bool,
    pub enable_sync_validation: bool,
    pub window_handle: Option<raw_window_handle::RawWindowHandle>,

    pub mesh_shaders: bool,
    pub atomic_float_operations: bool,
    pub ray_tracing: bool,
}

impl SgpuInititizationInfo {
    pub fn default_from_window<W: HasWindowHandle>(window: &W) -> SgpuInititizationInfo {
        SgpuInititizationInfo {
            window_handle: Some(window.window_handle().unwrap().as_raw()),
            ..Default::default()
        }
    }
}

impl Default for SgpuInititizationInfo {
    fn default() -> Self {
        Self {
            app_name: "Default",
            enable_validation_layers: true,
            enable_sync_validation: true,
            window_handle: None,
            mesh_shaders: false,
            atomic_float_operations: false,
            ray_tracing: false,
        }
    }
}

/// Main function, needs to be called before everythign else
pub fn sgpu_init(init_info: &SgpuInititizationInfo) { let _ = crate::CONTEXT.set(Context::new(init_info)); }

pub fn create_buffer(buffer_desc: &BufferDescription) -> Buffer { crate::CONTEXT.get().expect("Not initialized").create_buffer(buffer_desc) }

pub fn buffer_as_slice<T>(buffer: &Buffer) -> &[T] { crate::CONTEXT.get().expect("Not initialized").as_slice(buffer) }

pub fn buffer_as_slice_mut<T>(buffer: &Buffer) -> &mut [T] { crate::CONTEXT.get().expect("Not initialized").as_mut_slice(buffer) }

pub fn destroy_buffer(buffer: Buffer) { crate::CONTEXT.get().expect("Not initialized").destroy_buffer(buffer); }

pub fn create_image(image_desc: &ImageDescription) -> Image { crate::CONTEXT.get().expect("Not initialized").create_image(image_desc) }

// # Safety:
// Creates images and buffers that all share a single memory allocation.
pub unsafe fn create_aliased_resources(image_descs: &[ImageDescription], buffer_descs: &[BufferDescription]) -> (Vec<Image>, Vec<Buffer>) { crate::CONTEXT.get().expect("Not initialized").create_aliased_resources(image_descs, buffer_descs) }

pub fn destroy_image(image: Image) { crate::CONTEXT.get().expect("Not initialized").destroy_image(image); }

pub fn destroy_image_view(image_view: ImageView) { crate::CONTEXT.get().expect("Not initialized").destroy_image_view(image_view); }

pub fn create_sampler(sampler_desc: &SamplerDescription) -> Sampler { crate::CONTEXT.get().expect("Not initialized").create_sampler(sampler_desc) }

pub fn destroy_sampler(sampler: Sampler) { crate::CONTEXT.get().expect("Not initialized").destroy_sampler(sampler); }

pub fn destroy_rasterization_pipeline(pipeline: RasterizationPipeline) { crate::CONTEXT.get().expect("Not initialized").destroy_pipeline(pipeline.handle); }

pub fn destroy_compute_pipeline(pipeline: ComputePipeline) { crate::CONTEXT.get().expect("Not initialized").destroy_pipeline(pipeline.handle); }

pub fn create_swapchain<W: HasDisplayHandle + HasWindowHandle>(window: &W, swapchain_description: &SwapchainDescription) -> Swapchain {
    let ctx = crate::CONTEXT.get().expect("Not initialized");
    let surface = Surface::create_surface(&ctx.instance, window);
    let inner = InnerSwapchain::new(&surface, swapchain_description, None, ctx);

    Swapchain {
        inner,
        surface,
    }
}

pub fn create_rasterization_pipeline(raster_pipeline_desc: &RasterizationPipelineDescription) -> RasterizationPipeline { crate::CONTEXT.get().unwrap().create_raster_pipeline(raster_pipeline_desc) }

pub fn create_compute_pipeline(module: &ShaderModule) -> ComputePipeline { crate::CONTEXT.get().unwrap().create_compute_pipeline(module) }

pub fn create_query_pool(query_count: u32) -> QueryPool { crate::CONTEXT.get().expect("Not initialized").create_query_pool(query_count) }

/// A simple function to check if the counter has already been signaled
/// Not waits or anything
pub fn poll(counter: Counter) -> bool { crate::CONTEXT.get().expect("Not initialized").poll(counter) }

/// Wait for a counter to be signaled CPU side.
pub fn wait(counter: Counter) { crate::CONTEXT.get().expect("Not initialized").wait(counter); }

/// wait for all gpu work to be over
pub fn wait_idle() { crate::CONTEXT.get().expect("Not initialized").wait_idle(); }

/// began the recording of command buffer
pub fn record(queue_type: QueueType) -> CommandBuffer { crate::CONTEXT.get().expect("Not initialized").record(queue_type) }

/// submit command buffers
pub fn submit(command_buffers: &[CommandBuffer]) -> Counter { crate::CONTEXT.get().expect("Not initialized").submit(command_buffers) }
