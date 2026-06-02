mod api;
mod backend;
mod commands;
mod tests;
mod types;

use std::sync::OnceLock;

pub use api::*;
pub use commands::*;
pub use types::*;

static CONTEXT: OnceLock<backend::Context> = OnceLock::new();

// hmm
// buffers, textures, counters, fences, command buffers
// somehow i will need to manage all other shit internally
//
// i need to have a robust way to manage erros
//
// that was absent in nexion
