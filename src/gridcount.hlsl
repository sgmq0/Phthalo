
// RWStructuredBuffer<float4> output : register(u0);

// [numthreads(64, 1, 1)]
// void CSGridCount(uint3 DTid : SV_DispatchThreadID)
// {
//     output[DTid.x] = float4(DTid.x, 0, 0, 1);
// }

struct GPUParticle {
    float3 position;
    float _pad0;
    float3 predictedPosition;
    float _pad1;
    float3 velocity;
    float density;
    float lambda;
    int originalIndex;
    float _pad[2];
};

cbuffer NSConstants : register(b0) {
    float3 gridOrigin;
    float  cellSize;
    int3   gridDim;
    int    numParticles;
    int    numCells;
    float  dt;
    int2   _pad;
};

RWStructuredBuffer<int> cellCount            : register(u0);
RWStructuredBuffer<int> intraOffset          : register(u1);
RWStructuredBuffer<GPUParticle> particlesIn  : register(u2);
RWStructuredBuffer<int> cellStart            : register(u3);
RWStructuredBuffer<uint> statusBuf           : register(u4);
RWStructuredBuffer<GPUParticle> particlesOut : register(u5);

#define STATUS_SHIFT 30
#define VALUE_MASK   0x3FFFFFFF
#define TILE 256
groupshared int gs[TILE];
groupshared int gs_exclusivePrefix;

static const float H       = 1.0f;
static const float RHO_0   = 10.0f;
static const float EPSILON  = 100.0f;

float Poly6(float3 r, float h)
{
    float r2 = dot(r, r);
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float diff = h2 - r2;
    return (315.0f / (64.0f * 3.14159265f)) / pow(h, 9) * diff * diff * diff;
}

float3 SpikyGradient(float3 r, float h)
{
    float rLen = length(r);
    if (rLen > h || rLen < 1e-12f) return float3(0, 0, 0);
    float diff = h - rLen;
    float scalar = -(45.0f / (3.14159265f * pow(h, 6))) * diff * diff / rLen;
    return r * scalar;
}

[numthreads(64, 1, 1)]
void CSPrediction(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    // apply gravity
    particlesIn[i].velocity.y += -9.8f * dt;

    // predict position
    particlesIn[i].predictedPosition = particlesIn[i].position
        + dt * particlesIn[i].velocity;
}

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
            exclusivePrefix += value;

            if (status == 2) break;  // got a final inclusive prefix, stop
            lookIdx--;
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

    particlesIn[i].originalIndex = i;
    particlesOut[sortedIdx] = particlesIn[i];
}

[numthreads(64, 1, 1)]
void CSComputeLambda(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    //particlesIn[i].lambda = 999.0f;

    // particlesOut is sorted by cell — each thread handles one sorted slot
    GPUParticle pi = particlesOut[i];
    float3 pos_i = pi.predictedPosition;

    // compute cell of this particle
    int3 cell = clamp(
        (int3)floor((pos_i - gridOrigin) / cellSize),
        int3(0,0,0),
        gridDim - int3(1,1,1)
    );

    float density    = 0.0f;
    float3 gradSum_i = float3(0, 0, 0);  // sum of gradients w.r.t. i (self term)
    float denominator = 0.0f;
    float test = 5.0f;

    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++)
    {
        int3 nc = cell + int3(dx, dy, dz);
        if (any(nc < int3(0,0,0)) || any(nc >= gridDim)) continue;

        int flat  = nc.x + nc.y * gridDim.x + nc.z * gridDim.x * gridDim.y;
        int start = cellStart[flat];
        int count = cellCount[flat];

        for (int k = 0; k < count; k++)
        {
            float3 pos_j = particlesOut[start + k].predictedPosition;
            float3 r     = pos_i - pos_j;

            // density (Poly6)
            density += Poly6(r, H);

            // gradient of constraint w.r.t. k=j (eq. 8 denominator, k != i term)
            float3 grad_j = SpikyGradient(r, H) / RHO_0;
            denominator  += dot(grad_j, grad_j);

            // accumulate self-term gradient (k == i means sum over all j)
            gradSum_i += SpikyGradient(r, H);

            test += 1.0f;
        }
    }

    // add self-term (k == i) to denominator
    float3 grad_i_self = gradSum_i / RHO_0;
    denominator += dot(grad_i_self, grad_i_self);
    denominator += EPSILON;

    float constraint = (density / RHO_0) - 1.0f;

    // write to originalIndex so CPU-side m_particles[originalIndex] stays in sync
    particlesIn[i].lambda = -constraint / denominator;
    //particlesIn[i].lambda = test;
}
