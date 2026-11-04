use cargo_gpu_install::install::Install;
use cargo_gpu_install::spirv_builder::{Capability, ShaderPanicStrategy, SpirvMetadata};
use std::path::PathBuf;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_path = [
        manifest_dir,
        "shaders",
    ]
    .iter()
    .copied()
    .collect::<PathBuf>();

    let target = "spirv-unknown-vulkan1.3";
    let install = Install::from_shader_crate(crate_path.clone()).within_build_script().run()?;
    let mut builder = install.to_spirv_builder(crate_path, target);
    builder.capabilities.push(Capability::RuntimeDescriptorArray);
    builder.build_script.defaults = true;
    builder.shader_panic_strategy = ShaderPanicStrategy::SilentExit;
    builder.spirv_metadata = SpirvMetadata::Full;

    let compile_result = builder.build()?;
    let artifact = compile_result.module.unwrap_single();

    println!("cargo::rustc-env=SHADERS_SPV={}", artifact.display());
    Ok(())
}
