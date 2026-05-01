// constant buffer b0, updated every frame
// needs to match VPConstantBuffer in stdafx.h
cbuffer VPBuffer : register(b0)
{
    float4x4 vp;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : NORMAL;
};

PSInput VSMain(
    float4 position : POSITION, 
    float4 normal : NORMAL
) {
    PSInput result;
    result.position = mul(position, vp);  // clip space
    result.worldPos = position.xyz;       // world space
    result.normal = normal.xyz;
    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    float3 camPos2 = float3(1.0, 1.0, 1.0);
    float3 N = normalize(input.normal);
    float3 V = normalize(camPos2 - input.worldPos);

    float3 lightDir = normalize(float3(1.0, 2.0, 1.0));
    float3 lightColor = float3(1.0, 0.95, 0.85);
    
    float3 ambient = float3(0.05, 0.10, 0.18);

    // diffuse
    float diff = max(dot(N, lightDir), 0.0);
    float3 diffuse = diff * lightColor;

    float3 H = normalize(lightDir + V);
    float spec = pow(max(dot(N, H), 0.0), 128.0);
    float3 specular = spec * float3(1.0, 1.0, 1.0) * 0.8;

    // fresnel
    float  fresnel  = pow(1.0 - saturate(dot(N, V)), 4.0);
    float3 deepWater = float3(0.05, 0.15, 0.4);
    float3 shallowWater = float3(0.3,  0.7,  1.0);
    float3 waterColor = lerp(deepWater, shallowWater, fresnel);

    // rim lighting
    float rim = pow(1.0 - saturate(dot(N, V)), 3.0) * 0.3;
    float3 rimColor = float3(0.4, 0.7, 1.0) * rim;

    float3 color = (ambient + diffuse) * waterColor + specular + rimColor;

    return float4(color, 1.0);

    //return float4(0.3, 0.6, 1.0, 1.0);  // flat blue-ish water color
}