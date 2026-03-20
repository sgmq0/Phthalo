// constant buffer b0, updated every frame
cbuffer VPBuffer : register(b0)
{
    float4x4 vp;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
    float4 color : COLOR; 
    float3 worldPos : TEXCOORD0;
};

PSInput VSMain(
    float3 position : POSITION, 
    float3 normal : NORMAL, 
    float4 color : COLOR,
    float4 instanceRow0 : INSTANCE_TRANSFORM0,
    float4 instanceRow1 : INSTANCE_TRANSFORM1,
    float4 instanceRow2 : INSTANCE_TRANSFORM2,
    float4 instanceRow3 : INSTANCE_TRANSFORM3
)
{
    PSInput result;

    float4x4 worldMat = float4x4(instanceRow0, instanceRow1, instanceRow2, instanceRow3);
    float4 worldPos = mul(float4(position, 1.0f), worldMat);

    result.position = mul(worldPos, vp);
    result.normal = float4(normal, 1.0);
    result.color = color;
    result.worldPos = worldPos.xyz;

    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    // float3 norm = normalize(input.normal.xyz);
    // float3 lightPos = float3(10.0, -10.0, 10.0);
    // float3 lightDir = normalize(lightPos - input.worldPos);
    
    // float diff = max(dot(norm, lightDir), 0.0);
    // float3 result = diff * input.color.xyz;

    // return float4(result.x, result.y, result.z, 1.0);

    return float4(1.0, 0.0, 0.0, 1.0);
}

// compute shader
struct GPUParticle {
    float3 position;
    float  _pad0;
    float3 predictedPosition;
    float  _pad1;
    float3 velocity;
    float  density;
    float  lambda;
    float3 _pad2;
};

struct InstanceData {
    float4x4 worldMatrix;
};

cbuffer ComputeConstants : register(b1) {
    uint numParticles;
};

StructuredBuffer<GPUParticle>   particlesIn : register(t0);
RWStructuredBuffer<InstanceData> instancesOut : register(u0);

// [numthreads(64, 1, 1)]
// void CSUpdateInstances(uint3 id : SV_DispatchThreadID) {
//     uint i = id.x;
//     if (i >= numParticles) return;

//     float3 pos = particlesIn[i].position;

//     float4x4 mat = float4x4(
//         1, 0, 0, 0,
//         0, 1, 0, 0,
//         0, 0, 1, 0,
//         pos.x, pos.y, pos.z, 1
//     );

//     instancesOut[i] = mat;
// }
