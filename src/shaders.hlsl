// constant buffer b0, updated every frame
cbuffer MVPBuffer : register(b0)
{
    float4x4 mvp;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
    float4 color : COLOR;
};

PSInput VSMain(float4 position : POSITION, float4 normal : NORMAL, float4 color : COLOR)
{
    PSInput result;

    result.position = mul(position, mvp);
    result.normal = normal;
    result.color = color;

    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    float3 norm = normalize(input.normal.xyz);
    float3 lightPos = float3(10.0, -10.0, 10.0);
    float3 lightDir = normalize(lightPos - input.position.xyz);
    
    float diff = max(dot(norm, lightDir), 0.0);
    float3 result = diff * input.color.xyz;

    return float4(result.x, result.y, result.z, 1.0);
}
