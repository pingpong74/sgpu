struct VertexOutput {
    float4 position : SV_Position;
    float3 color    : COLOR;
};

static const float2 POSITIONS[3] = {
    float2( 0.0, -0.5),
    float2( 0.5,  0.5),
    float2(-0.5,  0.5),
};

static const float3 COLORS[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VertexOutput main(uint vertex_id : SV_VertexID) {
    VertexOutput out;
    out.position = float4(POSITIONS[vertex_id], 0.0, 1.0);
    out.color    = COLORS[vertex_id];
    return out;
}
