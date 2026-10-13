use sgpu_shaders::{
    SampledImageId,
    SamplerId,
    get_sampled_2d_f,
    glam::{Mat4, Vec2, Vec4},
    spirv,
};

#[repr(C)]
pub struct PushConstants {
    pub texture: SampledImageId,
    pub sampler: SamplerId,
    pub mvp: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

const VERTICES: [Vertex; 36] = [
    // +X
    Vertex { position: [0.5, -0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, 0.5, 0.5], uv: [1.0, 1.0] },
    // -X
    Vertex { position: [-0.5, -0.5, 0.5], uv: [0.0, 0.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [-0.5, 0.5, -0.5], uv: [1.0, 1.0] },
    // +Y
    Vertex { position: [-0.5, 0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, 0.5, 0.5], uv: [1.0, 1.0] },
    // -Y
    Vertex { position: [-0.5, -0.5, 0.5], uv: [0.0, 0.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, -0.5, -0.5], uv: [1.0, 1.0] },
    // +Z
    Vertex { position: [-0.5, -0.5, 0.5], uv: [0.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, 0.5, 0.5], uv: [1.0, 1.0] },
    Vertex { position: [-0.5, 0.5, 0.5], uv: [0.0, 1.0] },
    // -Z
    Vertex { position: [0.5, -0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], uv: [0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [-0.5, 0.5, -0.5], uv: [1.0, 1.0] },
];

#[spirv(vertex)]
pub fn cube_vs(#[spirv(vertex_index)] vertex_id: u32, #[spirv(position)] out_position: &mut Vec4, out_uv: &mut Vec2, #[spirv(push_constant)] push: &PushConstants) {
    let vertex = VERTICES[vertex_id as usize];
    let position = Vec4::new(vertex.position[0], vertex.position[1], vertex.position[2], 1.0);
    *out_position = Mat4::from_cols_array(&push.mvp) * position;
    *out_uv = Vec2::from_array(vertex.uv);
}

#[spirv(fragment)]
pub fn cube_fs(in_uv: Vec2, #[spirv(push_constant)] push: &PushConstants, output: &mut Vec4) {
    let image = get_sampled_2d_f(push.texture);
    let sampler = SamplerId::get_sampler(push.sampler);
    *output = image.sample(*sampler, in_uv);
}
