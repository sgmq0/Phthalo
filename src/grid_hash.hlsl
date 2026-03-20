struct GPUParticle {
    float3 position;
    float3 predictedPosition;
    float3 velocity;
    float  density;
    float  lambda;
};

cbuffer GridConstants : register(b0) {
    float3 gridOrigin;
    float  cellSize;
    int3   gridDim;
    int    numParticles;
};

StructuredBuffer<GPUParticle> particlesIn : register(t0);
RWStructuredBuffer<int>       cellCount  : register(u0);

int CellIndex(float3 pos)
{
    int3 cell = (int3)floor((pos - gridOrigin) / cellSize);
    cell = clamp(cell, int3(0, 0, 0), gridDim - 1);
    return cell.x
         + cell.y * gridDim.x
         + cell.z * gridDim.x * gridDim.y;
}

[numthreads(64, 1, 1)]
void CSGridHash(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= (uint)numParticles) return;
    int cell = CellIndex(particlesIn[tid.x].predictedPosition);
    InterlockedAdd(cellCount[cell], 1);
}
