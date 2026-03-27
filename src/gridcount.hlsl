
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
RWStructuredBuffer<GPUParticle> particlesIn : register(u2);
RWStructuredBuffer<int> cellStart : register(u3);
RWStructuredBuffer<uint> statusBuf : register(u4);
RWStructuredBuffer<GPUParticle> particlesOut : register(u5);

#define STATUS_SHIFT 30
#define VALUE_MASK   0x3FFFFFFF
#define TILE 256
groupshared int gs[TILE];
groupshared int gs_exclusivePrefix;

// step 1 - counting kernel

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

[numthreads(1, 1, 1)]
void CSBindingTest(uint3 tid : SV_DispatchThreadID) {
    cellCount[0] = 42;
}

[numthreads(64, 1, 1)]
void CSClearStatus(uint3 tid : SV_DispatchThreadID) {
    uint numGroups = ((uint)numCells + 255) / 256;
    if ((int)tid.x < numGroups)
        statusBuf[tid.x] = 0u;
}

[numthreads(TILE, 1, 1)]
void CSPrefixSum(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    /*
    idea: each thread posts its result to a shared array (u4, statusBuf), as soon as it's ready
    0: not started yet
    1: local aggregate ready (partial)
    2: inclusive prefix ready (final)
    [MG16] - 2016 paper by Merrill and Garland
    */

    uint global = gid.x * TILE + tid.x;
    uint prev;

    // 1. load tile from cellCount into shared memory
    if (tid.x == 0)
        gs_exclusivePrefix = 0;
    GroupMemoryBarrierWithGroupSync();

    gs[tid.x] = (global < (uint)numCells) ? cellCount[global] : 0;
    GroupMemoryBarrierWithGroupSync();

    // 2. local inclusive scan (Hillis-Steele)
    for (uint stride = 1; stride < TILE; stride <<= 1)
    {
        int val = (tid.x >= stride) ? gs[tid.x - stride] : 0;
        GroupMemoryBarrierWithGroupSync();
        gs[tid.x] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    // 3. write local aggregate with status = 1 (partial)
    if (tid.x == 0)
    {
        uint localSum = (uint) gs[TILE - 1];
        if (gid.x == 0)
        {
            InterlockedExchange(statusBuf[0], (2u << STATUS_SHIFT) | (localSum & VALUE_MASK), prev);
        }
        else
        {
            InterlockedExchange(statusBuf[gid.x], (1u << STATUS_SHIFT) | (localSum & VALUE_MASK), prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 4. look-back: thread 0 walks left accumulating the exclusive prefix
    if (tid.x == 0 && gid.x > 0)
    {
        int exclusivePrefix = 0;
        int lookIdx = (int)gid.x - 1;

        while (lookIdx >= 0)
        {
            uint packed;
            do {
                InterlockedAdd(statusBuf[lookIdx], 0u, packed);
            } while ((packed >> STATUS_SHIFT) == 0u);

            int status = packed >> STATUS_SHIFT;
            int value  = (int) (packed & VALUE_MASK);

            if (status == 2)
            {
                exclusivePrefix += value;
                break;
            }
            else
            {
                exclusivePrefix += value;
                lookIdx--;
            }
        }

        // write into the groupshared variable so all threads can read it
        gs_exclusivePrefix = exclusivePrefix;

        int myInclusive = exclusivePrefix + gs[TILE - 1];
        int prev;
        InterlockedExchange(statusBuf[gid.x],
            (2 << STATUS_SHIFT) | (myInclusive & VALUE_MASK), prev);
    }
    GroupMemoryBarrierWithGroupSync();

    // 5. all threads write their exclusive result to cellStart
    if (global < (uint)numCells)
    {
        int localExclusive = gs[tid.x] - cellCount[global];
        cellStart[global] = (gid.x == 0 ? 0 : gs_exclusivePrefix) + localExclusive;
    }
}

// step 3: counting kernel
[numthreads(64, 1, 1)]
void CSReorder(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    int cell      = CellIndex(particlesIn[i].predictedPosition);
    int sortedIdx = cellStart[cell] + intraOffset[i];

    particlesOut[sortedIdx] = particlesIn[i];
}

