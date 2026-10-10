use sgpu_shaders::{
    ReadOnlyBuffer,
    glam::{Vec3, Vec4, vec4},
    spirv,
};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PushConstants {
    pub vertex_buffer: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub _pad0: [f32; 2],
    pub color: [f32; 3],
    pub _pad1: f32,
}

#[spirv(vertex)]
pub fn triangle_vs(#[spirv(vertex_index)] vertex_id: u32, #[spirv(position)] out_position: &mut Vec4, out_color: &mut Vec3, #[spirv(push_constant)] push: &PushConstants) {
    let vertices = unsafe { ReadOnlyBuffer::<Vertex>::get(push.vertex_buffer) };
    let vertex = unsafe { vertices.load_indexed(vertex_id) };
    *out_position = vec4(vertex.position[0], vertex.position[1], 0.0, 1.0);
    *out_color = Vec3::from_array(vertex.color);
}

#[spirv(fragment)]
pub fn triangle_fs(in_color: Vec3, output: &mut Vec4) { *output = vec4(in_color.x, in_color.y, in_color.z, 1.0); }
