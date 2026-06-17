use crate::{Buffer, RasterizationPipeline};
use ash::vk;

pub struct RenderRecorder {
    pub(crate) cmd_buffer: vk::CommandBuffer,
}

impl RenderRecorder {
    pub fn bind_rasterization_pipeline(&self, pipeline: &RasterizationPipeline) {
        unsafe {
            crate::CONTEXT
                .get()
                .unwrap()
                .device
                .handle
                .cmd_bind_pipeline(self.cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
        }
    }

    pub fn push_constants<T: Copy>(&self, data: &T) {
        let ctx = crate::CONTEXT.get().unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>()) };
        unsafe {
            ctx.device
                .handle
                .cmd_push_constants(self.cmd_buffer, ctx.bindless_descriptor_set.pipeline_layout, vk::ShaderStageFlags::ALL, 0, bytes);
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            crate::CONTEXT
                .get()
                .unwrap()
                .device
                .handle
                .cmd_draw(self.cmd_buffer, vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    pub fn draw_indexed(&self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        unsafe {
            crate::CONTEXT
                .get()
                .unwrap()
                .device
                .handle
                .cmd_draw_indexed(self.cmd_buffer, index_count, instance_count, first_index, vertex_offset, first_instance);
        }
    }

    pub fn set_viewport(&self, width: u32, height: u32) {
        unsafe {
            crate::CONTEXT.get().unwrap().device.handle.cmd_set_viewport(
                self.cmd_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
        }
    }

    pub fn set_scissor(&self, width: u32, height: u32) {
        unsafe {
            crate::CONTEXT.get().unwrap().device.handle.cmd_set_scissor(
                self.cmd_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width,
                        height,
                    },
                }],
            );
        }
    }

    pub fn draw_indirect(&self, buffer: &Buffer, offset: u64, draw_count: u32, stride: u32) {
        let ctx = crate::CONTEXT.get().unwrap();
        unsafe {
            ctx.device.handle.cmd_draw_indirect(self.cmd_buffer, buffer.raw, offset, draw_count, stride);
        }
    }

    pub fn draw_indexed_indirect(&self, buffer: &Buffer, offset: u64, draw_count: u32, stride: u32) {
        let ctx = crate::CONTEXT.get().unwrap();
        unsafe {
            ctx.device.handle.cmd_draw_indexed_indirect(self.cmd_buffer, buffer.raw, offset, draw_count, stride);
        }
    }

    pub fn draw_indirect_count(&self, buffer: &Buffer, offset: u64, count_buffer: &Buffer, count_offset: u64, max_draw_count: u32, stride: u32) {
        let ctx = crate::CONTEXT.get().unwrap();
        unsafe {
            ctx.device
                .handle
                .cmd_draw_indirect_count(self.cmd_buffer, buffer.raw, offset, count_buffer.raw, count_offset, max_draw_count, stride);
        }
    }

    pub fn draw_indexed_indirect_count(&self, buffer: &Buffer, offset: u64, count_buffer: &Buffer, count_offset: u64, max_draw_count: u32, stride: u32) {
        let ctx = crate::CONTEXT.get().unwrap();
        unsafe {
            ctx.device
                .handle
                .cmd_draw_indexed_indirect_count(self.cmd_buffer, buffer.raw, offset, count_buffer.raw, count_offset, max_draw_count, stride);
        }
    }
}
