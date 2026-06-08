struct FragmentInput {
  float3 color : COLOR;
};

[shader("fragment")]
float4 main(FragmentInput in) : SV_Target { return float4(in.color, 1.0); }
