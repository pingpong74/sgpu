pub mod access;
mod graph;
mod node;
mod resource;

pub use graph::TaskGraph;
pub use node::{ExecutionContext, Node, NodeWriter};
pub use resource::{Id, PassBuffer, PassImage, PassView};

// TODO:
// Frames in flight
// Aliasing (Transient resources)
