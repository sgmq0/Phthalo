

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

[numthreads(64, 1, 1)]
void CSFinalize(uint3 tid : SV_DispatchThreadID)
{
    // int i = (int)tid.x;
    // if (i >= numParticles) return;

    // float3 pred = particlesIn[i].predictedPosition;
    // float3 pos = particlesIn[i].position;

    // // clamp to box
    // pred.x = clamp(pred.x, -boxSizeXZ, boxSizeXZ);
    // pred.y = clamp(pred.y, 0.0f, boxSizeY);
    // pred.z = clamp(pred.z, -boxSizeXZ, boxSizeXZ);

    // // derive velocity from displacement
    // float3 vel = damping * (pred - pos) / dt;

    // // kill velocity pointing into walls
    // if (pred.x <= -boxSizeXZ && vel.x < 0.0f) vel.x = 0.0f;
    // if (pred.x >=  boxSizeXZ && vel.x > 0.0f) vel.x = 0.0f;
    // if (pred.y <= 0.0f && vel.y < 0.0f) vel.y = 0.0f;
    // if (pred.z <= -boxSizeXZ && vel.z < 0.0f) vel.z = 0.0f;
    // if (pred.z >=  boxSizeXZ && vel.z > 0.0f) vel.z = 0.0f;

    // // apply XSPH
    // vel += float3(particlesIn[i].xsph) * viscosity;

    // particlesIn[i].predictedPosition = pred;
    // particlesIn[i].velocity = vel;
    // particlesIn[i].position = pred;
}

// from https://graphics.stanford.edu/~mdfisher/MarchingCubes.html
static const uint edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   
};
static const int triTable[256][16] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

[numthreads(1,1,1)]
void CSClearArgs()
{
    mcArgs[0] = 0;
}

[numthreads(64,1,1)]
void CSClearField(uint3 tid : SV_DispatchThreadID)
{
    uint total = (mcDim.x+1)*(mcDim.y+1)*(mcDim.z+1);
    if (tid.x < total) mcScalarField[tid.x] = 0.0f;
}

[numthreads(64,1,1)]
void CSBuildScalarField(uint3 tid : SV_DispatchThreadID)
{
    if ((int)tid.x >= numParticles) return;

    float3 pos = particlesIn[tid.x].position;

    // find the grid vertex this particle is closest to and splat
    int3 center = (int3)floor((pos - mcOrigin) / mcCellSize);

    for (int dx = -2; dx <= 2; dx++)
    for (int dy = -2; dy <= 2; dy++)
    for (int dz = -2; dz <= 2; dz++)
    {
        int3 gv = center + int3(dx, dy, dz);
        if (any(gv < int3(0,0,0)) ||
            any(gv > mcDim)) continue;   // note: mcDim, not mcDim-1

        float3 gvPos = mcOrigin + float3(gv) * mcCellSize;
        float3 r = pos - gvPos;
        int w = (int)(Poly6(r, H * 2.0f) * 100.0f);

        int idx = gv.x + gv.y*(mcDim.x+1) + gv.z*(mcDim.x+1)*(mcDim.y+1);
        InterlockedAdd(mcScalarField[idx], w);
    }
}

[numthreads(64,1,1)]
void CSMarchingCubes(uint3 tid : SV_DispatchThreadID)
{
    uint cellCount = (uint)(mcDim.x * mcDim.y * mcDim.z);
    if (tid.x >= cellCount) return;

    // unpack flat index -> 3D
    uint idx = tid.x;
    int3 c;
    c.x = idx % mcDim.x;
    c.y = (idx / mcDim.x) % mcDim.y;
    c.z = idx / (mcDim.x * mcDim.y);

    // 8 corner positions and values
    int3 corners[8] = {
        c+int3(0,0,0), c+int3(1,0,0), c+int3(1,0,1), c+int3(0,0,1),
        c+int3(0,1,0), c+int3(1,1,0), c+int3(1,1,1), c+int3(0,1,1)
    };

    float val[8];
    [unroll] for (int i = 0; i < 8; i++) {
        int3 v = corners[i];
        int fi = v.x + v.y*(mcDim.x+1) + v.z*(mcDim.x+1)*(mcDim.y+1);
        val[i] = mcScalarField[fi] / 100.0f;
        //val[i] = 1.0f;
    }

    // build lookup index
    uint cubeIdx = 0;
    [unroll] for (int i2 = 0; i2 < 8; i2++)
        if (val[i2] < mcIso) cubeIdx |= (1u << i2);

    if (edgeTable[cubeIdx] == 0) return;

    // interpolate edge vertices
    float3 worldCorners[8];
    [unroll] for (int i3 = 0; i3 < 8; i3++)
        worldCorners[i3] = mcOrigin + float3(corners[i3]) * mcCellSize;

    float3 edgeVerts[12];
    #define LERP_EDGE(e, a, b) \
        edgeVerts[e] = lerp(worldCorners[a], worldCorners[b], \
            (mcIso - val[a]) / (val[b] - val[a] + 1e-9f));

    if (edgeTable[cubeIdx] & 0x001) LERP_EDGE(0,  0, 1)
    if (edgeTable[cubeIdx] & 0x002) LERP_EDGE(1,  1, 2)
    if (edgeTable[cubeIdx] & 0x004) LERP_EDGE(2,  2, 3)
    if (edgeTable[cubeIdx] & 0x008) LERP_EDGE(3,  3, 0)
    if (edgeTable[cubeIdx] & 0x010) LERP_EDGE(4,  4, 5)
    if (edgeTable[cubeIdx] & 0x020) LERP_EDGE(5,  5, 6)
    if (edgeTable[cubeIdx] & 0x040) LERP_EDGE(6,  6, 7)
    if (edgeTable[cubeIdx] & 0x080) LERP_EDGE(7,  7, 4)
    if (edgeTable[cubeIdx] & 0x100) LERP_EDGE(8,  0, 4)
    if (edgeTable[cubeIdx] & 0x200) LERP_EDGE(9,  1, 5)
    if (edgeTable[cubeIdx] & 0x400) LERP_EDGE(10, 2, 6)
    if (edgeTable[cubeIdx] & 0x800) LERP_EDGE(11, 3, 7)

    // emit triangles
    for (int t = 0; triTable[cubeIdx][t] != -1; t += 3)
    {
        float3 v0 = edgeVerts[triTable[cubeIdx][t  ]];
        float3 v1 = edgeVerts[triTable[cubeIdx][t+1]];
        float3 v2 = edgeVerts[triTable[cubeIdx][t+2]];

        // compute flat normal
        float3 norm = normalize(cross(v1-v0, v2-v0));

        uint slot;
        InterlockedAdd(mcArgs[0], 3u, slot);
        if (slot + 2 < (uint)mcMaxTris * 3) {
            
            // encode normal later
            Vertex v0_vert = {v0.x, v0.y, v0.z, 1.0f, float4(norm, 1.0f)};
            Vertex v1_vert = {v1.x, v1.y, v1.z, 1.0f, float4(norm, 1.0f)};
            Vertex v2_vert = {v2.x, v2.y, v2.z, 1.0f, float4(norm, 1.0f)};

            mcVertexBuffer[slot] = v0_vert;
            mcVertexBuffer[slot+1] = v1_vert;
            mcVertexBuffer[slot+2] = v2_vert;
        }
    }
}

// one thread per voxel
// [numthreads(8, 8, 8)]
// void GenerateSDF(uint3 id : SV_DispatchThreadID)
// {
//     float3 voxelPos = GridToWorld(id); // convert voxel index to world space

//     float weightSum   = 0.0;
//     float3 weightedPos = float3(0, 0, 0);

//     // iterate over neighboring particles (from your neighbor search structure)
//     for each neighbor particle i within radius R:
//     {
//         float3 diff = voxelPos - particlePos[i];
//         float  d    = length(diff);

//         float t = 1.0 - (d * d) / (R * R);
//         float w = max(0.0, t * t * t);   // 6th order: (1 - d²/r²)³

//         weightSum    += w;
//         weightedPos  += particlePos[i] * w;
//     }

//     float sdf;
//     if (weightSum > 0.0)
//     {
//         weightedPos /= weightSum;
//         sdf = length(voxelPos - weightedPos) - particleRadius;
//     }
//     else
//     {
//         sdf = R; // no particles nearby, far outside surface
//     }

//     sdfVolume[id] = sdf;
// }