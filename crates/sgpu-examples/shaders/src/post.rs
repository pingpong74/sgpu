use sgpu_shaders::{
    SampledImageId, SamplerId, get_sampled_2d_f,
    glam::{Vec2, Vec3, Vec4},
    spirv,
};

#[repr(C)]
pub struct PushConstants {
    pub hdr: SampledImageId,
    pub sampler: SamplerId,
}

#[spirv(vertex)]
pub fn post_vs(#[spirv(vertex_index)] vertex_id: u32, out_uv: &mut Vec2, #[spirv(position)] out_position: &mut Vec4) {
    let uv = Vec2::new(((vertex_id << 1) & 2) as f32, (vertex_id & 2) as f32);
    *out_position = Vec4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
    *out_uv = uv;
}

#[spirv(fragment)]
pub fn post_fs(in_uv: Vec2, #[spirv(push_constant)] push: &PushConstants, output: &mut Vec4) {
    let image = get_sampled_2d_f(push.hdr);
    let sampler = SamplerId::get_sampler(push.sampler);
    let hdr = image.sample(*sampler, in_uv);
    let color = Vec3::new(hdr.x, hdr.y, hdr.z);
    let mapped = color / (color + Vec3::ONE);
    let display = mapped.powf(1.0 / 2.2);
    *output = Vec4::new(display.x, display.y, display.z, 1.0);
}
