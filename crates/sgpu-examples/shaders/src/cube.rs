use sgpu_shaders::{
    BufferId,
    ReadOnlyBuffer,
    Read,
    glam::{Mat4, Vec3, Vec4, vec4},
    spirv,
};

#[repr(C)]
pub struct PushConstants {
    pub vertex_buffer: BufferId<Read>,
    pub mvp: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[spirv(vertex)]
pub fn cube_vs(#[spirv(vertex_index)] vertex_id: u32, #[spirv(position)] out_position: &mut Vec4, out_color: &mut Vec3, #[spirv(push_constant)] push: &PushConstants) {
    let vertices = ReadOnlyBuffer::<Vertex>::get(push.vertex_buffer);
    let vertex = vertices.load_indexed(vertex_id);
    let position = Vec4::new(vertex.position[0], vertex.position[1], vertex.position[2], 1.0);
    *out_position = Mat4::from_cols_array(&push.mvp) * position;
    *out_color = Vec3::from_array(vertex.color);
}

#[spirv(fragment)]
pub fn cube_fs(in_color: Vec3, output: &mut Vec4) { *output = vec4(in_color.x, in_color.y, in_color.z, 1.0); }
