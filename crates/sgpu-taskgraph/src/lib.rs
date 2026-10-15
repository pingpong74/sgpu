//! A minimal, allocation-free-per-frame task graph for `sgpu`.
//!
//! Add external resources ([`Id`]s) and nodes ([`Node`]) to a [`Graph`], then
//! run it every frame. Barriers and cross-queue waits are synthesized from the
//! accesses nodes declare at compile time, and replayed per frame: cold chains
//! on the first execute, hot chains (ordered against the previous frame) after.

mod access;
mod graph;
mod node;
mod resource;

pub use access::*;
pub use graph::Graph;
pub use node::{ExecutionContext, Node, NodeWriter};
pub use resource::{Id, PassBuffer, PassImage};
