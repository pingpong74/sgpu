mod api;
mod backend;
mod commands;
mod swapchain;
mod tests;
mod types;

use std::sync::OnceLock;

pub use api::*;
pub use commands::*;
pub use swapchain::*;
pub use types::*;

static CONTEXT: OnceLock<backend::Context> = OnceLock::new();

// hmm
