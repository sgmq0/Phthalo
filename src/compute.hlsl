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

cbuffer ComputeConstants : register(b0) {
    uint numParticles;
};

StructuredBuffer<GPUParticle>    particlesIn  : register(t0);
RWStructuredBuffer<InstanceData> instancesOut : register(u0);

[numthreads(64, 1, 1)]
void CSUpdateInstances(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= numParticles) return;

    float3 pos = particlesIn[i].position;

    float4x4 mat = float4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos.x, pos.y, pos.z, 1
    );

    instancesOut[i] = mat;
}
