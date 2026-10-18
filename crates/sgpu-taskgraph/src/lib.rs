pub mod access;
mod graph;
mod node;
mod resource;

pub use graph::TaskGraph;
pub use node::{ExecutionContext, Node, NodeWriter};
pub use resource::{Id, PassBuffer, PassImage};

// TODO:
// Better ImageView handling
// Frames in flight
// Some better means to resize resources
// Better indices
// Aliasing (Transient resources)
