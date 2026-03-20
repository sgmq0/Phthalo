struct GridEntry
{
    uint cellIndex;
    uint particleIndex;
};

RWStructuredBuffer<float4>    positions   : register(u0);
RWStructuredBuffer<GridEntry> gridEntries : register(u1);

cbuffer SimConstants : register(b0)
{
    float cellSize;
    int   gridDim;
    float gridOrigin;
    int   numParticles;
}

[numthreads(64, 1, 1)]
void CSAssign(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= (uint)numParticles) return;

    float3 pos = positions[i].xyz;

    int cx = (int)floor((pos.x - gridOrigin) / cellSize);
    int cy = (int)floor((pos.y - gridOrigin) / cellSize);
    int cz = (int)floor((pos.z - gridOrigin) / cellSize);

    cx = clamp(cx, 0, gridDim - 1);
    cy = clamp(cy, 0, gridDim - 1);
    cz = clamp(cz, 0, gridDim - 1);

    gridEntries[i].cellIndex     = cx + cy * gridDim + cz * gridDim * gridDim;
    gridEntries[i].particleIndex = i;
}