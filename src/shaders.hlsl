// constant buffer b0, updated every frame
cbuffer VPBuffer : register(b0)
{
    float4x4 vp;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
};

PSInput VSMain(
    float4 position : POSITION, 
    float4 normal : NORMAL
) {
    PSInput result;
    result.position = mul(position, vp);
    result.normal = normal;
    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    float3 norm = normalize(input.normal.xyz);
    float3 lightPos = float3(10.0, 10.0, 10.0);
    float3 lightDir = normalize(lightPos - input.position.xyz);
    
    float diff = max(dot(norm, lightDir), 0.0);

    float4 water_color = float4(0.3, 0.6, 1.0, 1.0);
    float3 result = diff * water_color;

    return float4(result.x, result.y, result.z, 1.0);

    //return float4(0.3, 0.6, 1.0, 1.0);  // flat blue-ish water color
}