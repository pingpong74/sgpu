import sgpu;

struct VertexOutput {
  float4 position : SV_Position;
  float3 color : COLOR;
};

struct Vertex {
  float2 position;
  float2 _pad1;
  float3 color;
  float _pad2;
};

[[vk_push_constant]]
uint vertex_buffer_handle;

static const float2 POSITIONS[3] = {
  float2(0.0, -0.5),
  float2(0.5, 0.5),
  float2(-0.5, 0.5),
};

static const float3 COLORS[3] = {
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VertexOutput main(uint vertex_id: SV_VertexID) {
  ReadOnlyBuffer<Vertex> vertex_buffer =
      get_buffer<Vertex>(vertex_buffer_handle);

  VertexOutput out;

  let vertex = vertex_buffer[vertex_id];
  out.position = float4(vertex.position, 0.0, 1.0);
  out.color = vertex.color;
  return out;
}
