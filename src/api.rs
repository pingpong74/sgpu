use crate::{backend::Context, types::*};

pub struct SgpuInititizationInfo {
    pub app_name: &'static str,
    pub enable_validation_layers: bool,
    pub window_handle: Option<raw_window_handle::RawWindowHandle>,

    pub mesh_shaders: bool,
    pub atomic_float_operations: bool,
    pub ray_tracing: bool,
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

/// Create buffer, not thread safe ( For now ?)
pub fn create_buffer(buffer_desc: &BufferDescription) -> Buffer {
    return crate::CONTEXT.get().expect("Not initialized").create_buffer(buffer_desc);
}

/// Destroy buffer, not thread safe?
pub fn destroy_buffer(buffer: Buffer) {
    crate::CONTEXT.get().expect("Not initialized").destroy_buffer(buffer);
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
