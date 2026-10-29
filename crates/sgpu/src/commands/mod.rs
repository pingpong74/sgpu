mod barrier;
mod command_buffer;
mod rendering_recorder;

pub use barrier::*;
pub use command_buffer::*;
pub use rendering_recorder::*;

/// draw indirect command, for direct use
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}
