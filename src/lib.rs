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

pub fn add_shader_directory(path: &str) {
    let dir = std::path::Path::new(path);
    if !dir.exists() {
        println!("Directory provied doesn't exist, attempting to create the directory");
        std::fs::create_dir_all(dir).expect("Failed to create the directory");
    }

    let output = dir.join("sgpu.slang");
    std::fs::write(output, include_bytes!("sgpu.slang")).expect("Failed to write nexion.slang to the requested directory");
}

// hmm
