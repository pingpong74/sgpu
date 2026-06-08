mod api;
mod backend;
mod commands;
mod pipeline;
mod swapchain;
mod tests;
mod types;

use std::sync::OnceLock;

pub use api::*;
pub use commands::*;
pub use pipeline::*;
pub use swapchain::*;
pub use types::*;

static CONTEXT: OnceLock<backend::Context> = OnceLock::new();

// hmm
