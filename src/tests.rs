#[cfg(test)]
mod tests {
    use crate::{BufferDescription, BufferUsage, SgpuInititizationInfo};

    #[test]
    fn test() {
        crate::sgpu_init(&SgpuInititizationInfo {
            app_name: "Test app",
            enable_validation_layers: true,
            window_handle: None,
            mesh_shaders: false,
            atomic_float_operations: false,
            ray_tracing: false,
        });

        let buffer = crate::create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE,
            size: 1000,
            memory_type: crate::MemoryType::HostVisible,
        });

        crate::destroy_buffer(buffer);
    }
}
