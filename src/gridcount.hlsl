
RWStructuredBuffer<float4> output : register(u0);

[numthreads(64, 1, 1)]
void CSGridCount(uint3 DTid : SV_DispatchThreadID)
{
    output[DTid.x] = float4(DTid.x, 0, 0, 1);
}
