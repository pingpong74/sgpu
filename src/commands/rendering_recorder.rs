use ash::vk;

pub struct RenderRecorder {
    pub(crate) cmd_buffer: vk::CommandBuffer,
}

impl RenderRecorder {
    pub fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            crate::CONTEXT
                .get()
                .unwrap()
                .device
                .handle
                .cmd_draw(self.cmd_buffer, vertex_count, instance_count, first_vertex, first_instance);
        };
    }
}
