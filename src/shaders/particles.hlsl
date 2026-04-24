struct GPUParticle {
    float3 position;
    float _pad0;
    float3 predictedPosition;
    float _pad1;
    float3 velocity;
    float density;
    float lambda;
    int originalIndex;
    float2 _pad2;
    float3 xsph;
    float _pad3;
    float3 delta;
    float _pad4;
};

struct Vertex {
    float x;
    float y; 
    float z; 
    float w;
    float4 normal;
};

cbuffer NSConstants : register(b0) {
    float3 gridOrigin;  // bottom right of the bounding box
    float cellSize;     // size of each of the boxes, this is the same as H
    int3 gridDim;       // how many cells in each dimension
    int numParticles;   // number of particles in total
    float3 size;        // size of the bounding box
    float _pad;
    int numCells;       // number of cells overall
    float dt;           // just the timestep
    float H;            // smoothing
    float RHO_0;        // rest density
    float EPSILON;
    float3 _pad1;
};

cbuffer MCConstants : register(b1) {
    float3 mcOrigin;
    float mcCellSize;
    int3 mcDim;
    float mcIso;
    int mcMaxTris;
    float3 _pad2;
}

RWStructuredBuffer<int> cellCount               : register(u0);
RWStructuredBuffer<int> intraOffset             : register(u1);
RWStructuredBuffer<GPUParticle> particlesIn     : register(u2);
RWStructuredBuffer<int> cellStart               : register(u3);
RWStructuredBuffer<uint> statusBuf              : register(u4); // buffer for prefix sum algo    
RWStructuredBuffer<GPUParticle> particlesOut    : register(u5);

RWStructuredBuffer<int> mcScalarField         : register(u6);
RWStructuredBuffer<Vertex> mcVertexBuffer       : register(u7); // needs to match Vertex in stdafx.h
RWStructuredBuffer<uint> mcArgs                 : register(u8); 
RWStructuredBuffer<float> sdfVolume        : register(u9);

// values for uniform grid search
#define STATUS_SHIFT 30
#define VALUE_MASK   0x3FFFFFFF
#define TILE 256
groupshared int gs[TILE];
groupshared int gs_exclusivePrefix;

#define MU_S 0.1f
#define MU_K 0.05f

float Poly6(float3 r, float h)
{
    float coeff = 315.0 / (64.0 * 3.14159265f);

    float h2 = h * h;
    float r2 = dot(r, r);

    if (r2 > h2) {
        return 0.0;
    }

    float h9 = pow(abs(h), 9);
    float diff = h2 - r2;
    float diff3 = diff * diff * diff;

    return (coeff / h9) * diff3;
}

float3 SpikyGradient(float3 r, float h)
{
    float norm = length(r);

    if (norm > h || norm < 1e-12f) return float3(0, 0, 0);

    float h6 = pow(h, 6);
    float diff = h - norm;
    float diff2 = diff * diff;

    float scalar = -(45.0f / (3.14159265f * h6)) * diff2 / norm;
    return r * scalar;
}

// Zhu-Bridson 6th order smooth kernel
float ZBKernel(float r, float h)
{
    if (r >= h) return 0.0f;
    float t = r / h;
    float x = 1.0f - t * t;
    return x * x * x;
}

// ----------------- UNIFORM GRID SEARCH KERNELS --------------------

// helper functions here
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
void CSClearStatus(uint3 tid : SV_DispatchThreadID) {
    uint numGroups = ((uint)numCells + 255) / 256;
    if (tid.x < numGroups)
        statusBuf[tid.x] = 0u;
}

// step 1 - counting kernel
// determines the indexing offset within a cell. (intra-cell offset)
[numthreads(64, 1, 1)]
void CSCounting(uint3 tid : SV_DispatchThreadID) {
    if ((int)tid.x >= numParticles) return;
    int cell = CellIndex(particlesIn[tid.x].predictedPosition);
    int slot;
    InterlockedAdd(cellCount[cell], 1, slot);
    intraOffset[tid.x] = slot;
}

// step 2 - performs prefix sum operation. 
// this is the indexing offset between different cells. (inter-cell offset)
// loads into cellStart
// based on [MG16] - 2016 paper by Merrill and Garland
[numthreads(TILE, 1, 1)]
void CSPrefixSum(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    /*
    idea: each thread posts its result to a shared array (u4, statusBuf), as soon as it's ready
    0: not started yet
    1: local aggregate ready (partial)
    2: inclusive prefix ready (final)
    ngl i don't fully understand this
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

// step 3: reordering kernel
// simple sort on the particle data.
[numthreads(64, 1, 1)]
void CSReorder(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    int cell = CellIndex(particlesIn[i].predictedPosition);
    int sortedIdx = cellStart[cell] + intraOffset[i];

    particlesIn[i].originalIndex = i;
    particlesOut[sortedIdx] = particlesIn[i];
}

// ----------------- PBF SIMULATION KERNELS --------------------

// step 1: predict positions
[numthreads(64, 1, 1)]
void CSPrediction(uint3 tid : SV_DispatchThreadID)
{
    int i = (int) tid.x;
    if (i >= numParticles) return;

    // apply forces
    //particlesIn[i].velocity.y += -9.8f * dt;
    particlesIn[i].velocity.y += -1.0f * dt;

    // predict position
    particlesIn[i].predictedPosition = particlesIn[i].position + dt * particlesIn[i].velocity;
}

// step 2 is neighbor search, dispatched from ParticleSystem::DispatchGPUCommands

// step 3: find lambda_i
[numthreads(64, 1, 1)]
void CSComputeLambda(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    // go in order of position/cell (sorted in particlesOut) instead of unsorted (particlesIn)
    GPUParticle pi = particlesOut[i];
    float3 pos_i = pi.predictedPosition;

    // compute cell of this particle
    int3 cell = clamp(
        (int3)floor((pos_i - gridOrigin) / cellSize),
        int3(0,0,0),
        gridDim - int3(1,1,1)
    );

    // calc constraint
    float density = 0.0;
    float denominator = 0.0;
    float3 gradSum = float3(0, 0, 0);

    // for neighbors of i (current particle):
    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++)
    {
        int3 nc = cell + int3(dx, dy, dz);
        if (any(nc < int3(0,0,0)) || any(nc >= gridDim)) continue;

        int flat = nc.x + nc.y * gridDim.x + nc.z * gridDim.x * gridDim.y;
        int start = cellStart[flat];
        int count = cellCount[flat];

        for (int k = 0; k < count; k++)
        {
            int j = start + k;
            float3 pos_j = particlesOut[j].predictedPosition;
            float3 r = pos_i - pos_j;
            
            // denominator calcs
            float3 grad = -SpikyGradient(pos_i - pos_j, H) / RHO_0; // calc grad constraint
            denominator += dot(grad, grad);

            // numerator calcs
            density += Poly6(pos_i - pos_j, H);

            // accumulate self term
            gradSum += SpikyGradient(pos_i - pos_j, H);
        }
    }
    
    float numerator = (density / RHO_0) - 1.0f;
    float3 gradSumFinalized = gradSum / RHO_0;
    denominator += dot(gradSumFinalized, gradSumFinalized);
    denominator += EPSILON;

    // store lambda
    particlesIn[pi.originalIndex].lambda = -numerator / denominator;
}

[numthreads(64, 1, 1)]
void CSComputeDelta(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    GPUParticle pi = particlesOut[i];  // sorted slot i
    float3 pos_i = pi.predictedPosition;
    float lambda_i = particlesIn[pi.originalIndex].lambda;  // written by CSComputeLambda

    int3 cell = clamp(
        (int3)floor((pos_i - gridOrigin) / cellSize),
        int3(0, 0, 0),
        gridDim - int3(1, 1, 1)
    );

    // tensile correction constants
    const float corr_n = 4.0f;
    const float corr_h = 0.30f;
    const float corr_k = 1e-02f;
    const float corr_w = Poly6(float3(corr_h * H, 0.0f, 0.0f), H);

    // calculate delta p
    float3 delta = float3(0, 0, 0);

    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++)
    {
        int3 nc = cell + int3(dx, dy, dz);
        if (any(nc < int3(0,0,0)) || any(nc >= gridDim)) continue;

        int flat = nc.x + nc.y * gridDim.x + nc.z * gridDim.x * gridDim.y;
        int start = cellStart[flat];
        int count = cellCount[flat];

        for (int k = 0; k < count; k++)
        {
            int j = start + k;
            GPUParticle pj = particlesOut[start + k];
            float3 pos_j = pj.predictedPosition;
            float lambda_j = particlesIn[pj.originalIndex].lambda;

            float3 r = pos_i - pos_j;
            float poly = Poly6(r, H);
            float ratio = (corr_w > 1e-12f) ? (poly / corr_w) : 0.0f;
            float corr = -corr_k * pow(ratio, corr_n);
            float coeff = lambda_i + lambda_j + corr;
            float3 grad = coeff * SpikyGradient(r, H);

            float lambda_sum = lambda_i + lambda_j;

            delta += grad;
        }
    }

    delta /= RHO_0;

    // accumulate into predictedPosition directly
    particlesIn[pi.originalIndex].delta = delta;
}

[numthreads(64, 1, 1)]
void CSCollisionConstraints(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    GPUParticle pi = particlesIn[i];
    int oi = pi.originalIndex;

    float particleRadius = 0.15f;
    float3 posMin = float3(-size.x + particleRadius, particleRadius, -size.z + particleRadius);
    float3 posMax = float3(size.x - particleRadius, size.y - particleRadius, size.z - particleRadius);

    float3 oldPos = particlesIn[oi].predictedPosition;
    float3 newPos = oldPos + particlesIn[oi].delta;
    newPos = clamp(newPos, posMin, posMax);

    particlesIn[oi].predictedPosition = newPos;
    //particlesIn[oi].delta = float3(0, 0, 0);
}

[numthreads(64, 1, 1)]
void CSUpdateVelocity(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    GPUParticle pi = particlesOut[i];  // sorted slot i
    int orig = pi.originalIndex;

    float3 pos  = particlesIn[orig].position;
    float3 pred = particlesIn[orig].predictedPosition;

    const float DAMPING = 0.999f;
    float3 vel = DAMPING * (pred - pos) / dt;

    particlesOut[orig].velocity = vel;
}

[numthreads(64, 1, 1)]
void CSComputeXSPH(uint3 tid : SV_DispatchThreadID)
{
    int i = (int)tid.x;
    if (i >= numParticles) return;

    // particlesOut is sorted, particlesIn is original-order
    // we iterate over sorted neighbors via the grid, same as before
    GPUParticle pi = particlesOut[i];
    float3 pos_i = pi.predictedPosition;
    float3 vel_i = pi.velocity;

    int3 cell = clamp(
        (int3) floor((pos_i - gridOrigin) / cellSize),
        int3(0, 0, 0),
        gridDim - int3(1, 1, 1)
    );

    // 1: compute density
    float density = 0.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++)
            {
                int3 nc = cell + int3(dx, dy, dz);
                if (any(nc < int3(0,0,0)) || any(nc >= gridDim)) continue;
                int flat = nc.x + nc.y * gridDim.x + nc.z * gridDim.x * gridDim.y;
                int start = cellStart[flat];
                int cellN = cellCount[flat];
                for (int k = 0; k < cellN; k++)
                {
                    float3 r = pos_i - particlesOut[start + k].predictedPosition;
                    density += Poly6(r, H);
                }
            }
        }
    }

    // guard against zero density
    float invDensity = (density > 1e-6f) ? (1.0f / density) : 0.0f;

    // 2: XSPH viscosity
    float3 pos_i_cur = particlesIn[pi.originalIndex].position;

    float3 xsph = float3(0, 0, 0);

    for (int dx2 = -1; dx2 <= 1; dx2++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int3 nc = cell + int3(dx2, dy, dz);
                if (any(nc < int3(0,0,0)) || any(nc >= gridDim)) continue;
                int flat  = nc.x + nc.y * gridDim.x + nc.z * gridDim.x * gridDim.y;
                int start = cellStart[flat];
                int cellN = cellCount[flat];
                for (int k = 0; k < cellN; k++)
                {
                    GPUParticle pj = particlesOut[start + k];
                    float3 pos_j_cur = particlesIn[pj.originalIndex].position;

                    float3 r = pos_i_cur - pos_j_cur;
                    float val = Poly6(r, H);

                    float3 vel_j = pj.velocity;
                    xsph += invDensity * val * (vel_j - vel_i);
                }
            }
        }
    }

    // write xsph 
    particlesIn[pi.originalIndex].density = density;
    particlesIn[pi.originalIndex].xsph = xsph;
}
