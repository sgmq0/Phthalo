
// RWStructuredBuffer<float4> output : register(u0);

// [numthreads(64, 1, 1)]
// void CSGridCount(uint3 DTid : SV_DispatchThreadID)
// {
//     output[DTid.x] = float4(DTid.x, 0, 0, 1);
// }

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

cbuffer NSConstants : register(b0) {
    float3 gridOrigin;
    float  cellSize;
    int3   gridDim;
    int    numParticles;
    int    numCells;
    int3   _pad;
};

RWStructuredBuffer<int>        cellCount   : register(u0);
RWStructuredBuffer<int>        intraOffset : register(u1);
StructuredBuffer<GPUParticle>  particlesIn : register(t0);

int CellIndex(float3 pos) {
    int3 cell = (int3)floor((pos - gridOrigin) / cellSize);
    cell = clamp(cell, int3(0,0,0), gridDim - int3(1,1,1));
    return cell.x
         + cell.y * gridDim.x
         + cell.z * gridDim.x * gridDim.y;
}

[numthreads(64, 1, 1)]
void CSClearCells(uint3 tid : SV_DispatchThreadID) {
    if ((int)tid.x < numCells)
        cellCount[tid.x] = 0;
}

[numthreads(64, 1, 1)]
void CSCounting(uint3 tid : SV_DispatchThreadID) {
    if ((int)tid.x >= numParticles) return;
    int cell = CellIndex(particlesIn[tid.x].predictedPosition);
    int slot;
    InterlockedAdd(cellCount[cell], 1, slot);
    intraOffset[tid.x] = slot;
}
