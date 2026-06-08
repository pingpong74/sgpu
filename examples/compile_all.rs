use std::{fs, path::Path, process::Command};

// compiles all shaders for all exmaples

fn main() {
    compile_directory(Path::new("examples/shaders"));
}

fn compile_directory(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            compile_directory(&path);
            continue;
        }

        if path.extension().and_then(|s| s.to_str()) != Some("slang") {
            continue;
        }

        if path.file_name().and_then(|s| s.to_str()) == Some("sgpu.slang") {
            continue;
        }

        compile_shader(&path);
    }
}

fn compile_shader(path: &Path) {
    let output = path.with_extension("spv");

    println!("Compiling {} -> {}", path.display(), output.display());

    let status = Command::new("slangc").arg(path).arg("-o").arg(&output).status().expect("Failed to launch slangc");

    assert!(status.success(), "Shader compilation failed: {}", path.display());
}
