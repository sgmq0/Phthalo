// constant buffer b0, updated every frame
cbuffer VPBuffer : register(b0)
{
    float4x4 vp;
};

struct PSInput
{
    float4 position : SV_POSITION;
};

PSInput VSMain(float4 position : POSITION)
{
    PSInput result;
    result.position = mul(position, vp);
    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return float4(0.3, 0.6, 1.0, 1.0);  // flat blue-ish water color
}