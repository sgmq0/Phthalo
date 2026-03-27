#include "stdafx.h"
#include "ParticleSystem.h"
#include <iostream>
#include <algorithm>

ParticleSystem::ParticleSystem() :
    m_instancer(Instancer())
{
}

ParticleSystem::ParticleSystem(UINT numParticles) :
    m_instancer(Instancer())
{  
}

void ParticleSystem::LoadParticles()
{
	// initializes all the particles
	float spacing = 0.3f;
    int perAxis = (int)cbrt(NUM_PARTICLES);
    int i = 0;
    for (int x = 0; x < perAxis; x++)
    for (int y = 0; y < perAxis; y++)
    for (int z = 0; z < perAxis; z++) {
        if (i >= NUM_PARTICLES) break;
        m_particles[i].position = {
            (x - perAxis/2.0f) * spacing,
            y * spacing + 1.0f,   // start above floor
            (z - perAxis/2.0f) * spacing
        };
        m_particles[i].velocity = {0, 0, 0};
        m_particles[i].predictedPosition = m_particles[i].position; 
        i++;
    }
    
    m_instancer.m_instances.resize(NUM_PARTICLES);
}

ComPtr<ID3DBlob> CompileHelper(std::wstring shaderPath, const char* entry) {
    ComPtr<ID3DBlob> computeShader, computeError;
    HRESULT hr = D3DCompileFromFile(shaderPath.c_str(),
        nullptr, nullptr, entry, "cs_5_0", 0, 0, &computeShader, &computeError);
    if (computeError) OutputDebugStringA((char*)computeError->GetBufferPointer());
    ThrowIfFailed(hr);
    return computeShader;
};

ComPtr<ID3D12PipelineState> MakePSOHelper(ComPtr<ID3DBlob> computeShader, ComPtr<ID3D12RootSignature> rootSignature, ID3D12Device* device) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    ComPtr<ID3D12PipelineState> pso;
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));
    return pso;
};

ComPtr<ID3D12Resource> MakeBufferHelper(UINT byteSize, ID3D12Device* device) {
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

    ComPtr<ID3D12Resource> buffer;
    ThrowIfFailed(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
        &bufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&buffer)));
    return buffer;
}

void ParticleSystem::CreateComputePipeline(
    ID3D12Device *device,
    std::wstring shaderPath,
    ComPtr<ID3D12CommandAllocator> &commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> &commandList)
{
    // 1. root signature

    CD3DX12_ROOT_PARAMETER params[7];
    params[0].InitAsConstantBufferView(0);  // b0: NSConstants
    params[1].InitAsUnorderedAccessView(0); // u0: cellCount
    params[2].InitAsUnorderedAccessView(1); // u1: intraOffset
    params[3].InitAsUnorderedAccessView(2);  // u2: particlesIn
    params[4].InitAsUnorderedAccessView(3); // u3: cellStart
    params[5].InitAsUnorderedAccessView(4); // u4: groupSums
    params[6].InitAsUnorderedAccessView(5); // u5: particlesOut

    CD3DX12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = 7;
    rootDesc.pParameters = params;
    rootDesc.NumStaticSamplers = 0;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(),
        sig->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));

    // 2/3. compile shaders & pipeline states
    ComPtr<ID3DBlob> prediction = CompileHelper(shaderPath, "CSPrediction");
    m_psoPrediction = MakePSOHelper(prediction.Get(), m_computeRootSignature.Get(), device);

    ComPtr<ID3DBlob> clearCells = CompileHelper(shaderPath, "CSClearCells");
    ComPtr<ID3DBlob> count = CompileHelper(shaderPath, "CSCounting");
    ComPtr<ID3DBlob> clearStatus = CompileHelper(shaderPath, "CSClearStatus");
    ComPtr<ID3DBlob> prefixSum = CompileHelper(shaderPath, "CSPrefixSum");
    ComPtr<ID3DBlob> reorder = CompileHelper(shaderPath, "CSReorder");
    m_psoClear = MakePSOHelper(clearCells.Get(), m_computeRootSignature.Get(), device);
    m_psoCount = MakePSOHelper(count.Get(), m_computeRootSignature.Get(), device);
    m_psoClearStatus = MakePSOHelper(clearStatus.Get(), m_computeRootSignature.Get(), device);
    m_psoPrefixScanPass = MakePSOHelper(prefixSum.Get(), m_computeRootSignature.Get(), device);
    m_psoReorder = MakePSOHelper(reorder.Get(), m_computeRootSignature.Get(), device);

    ComPtr<ID3DBlob> computeLambda = CompileHelper(shaderPath, "CSComputeLambda");
    m_psoComputeLambda = MakePSOHelper(prediction.Get(), m_computeRootSignature.Get(), device);

    // 4. all of the buffers
    m_nsParticlesIn = MakeBufferHelper(NUM_PARTICLES * sizeof(GPUParticle), device);
    m_nsCellCount = MakeBufferHelper(NS_NUM_CELLS * sizeof(int), device);
    m_nsIntraOffset = MakeBufferHelper(NUM_PARTICLES * sizeof(int), device);
    m_nsCellStart = MakeBufferHelper(NS_NUM_CELLS * sizeof(int), device);
    UINT numGroups = (NS_NUM_CELLS + 255) / 256;
    m_nsStatusBuf = MakeBufferHelper(numGroups * sizeof(int), device);
    m_nsParticlesOut = MakeBufferHelper(NUM_PARTICLES * sizeof(GPUParticle), device);

    // upload buffer, which CPU writes to, then CopyResource to m_nsParticlesIn
    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            NUM_PARTICLES * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_nsUploadBuffer)));
    }

    // const buffer (256-byte aligned)
    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        UINT sz = (sizeof(NSConstants) + 255) & ~255;
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_nsConstantBuffer)));
    }

    // readback buffers
    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(NS_NUM_CELLS * sizeof(int));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_nsReadbackCellCount)));
    }

    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(NS_NUM_CELLS * sizeof(int));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_nsReadbackCellStart)));
    }

    {
    CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(NUM_PARTICLES * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_nsReadbackParticlesOut)));
    }

    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(NUM_PARTICLES * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_nsReadbackParticlesIn))); // new ComPtr member
    }

    // 5. command allocator + list
    ThrowIfFailed(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&commandAllocator)));
    ThrowIfFailed(device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        commandAllocator.Get(),
        m_psoClear.Get(),
        IID_PPV_ARGS(&commandList)));
    ThrowIfFailed(commandList->Close());

    std::cout << "compute pipeline created" << std::endl;
}

void ParticleSystem::DispatchGPUCommands(ID3D12GraphicsCommandList *cmdList, float dt)
{
    DispatchInit(cmdList, dt);
    DispatchPrediction(cmdList, dt);
    DispatchNeighborSearch(cmdList);

    cmdList->SetPipelineState(m_psoComputeLambda.Get());
    cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
    auto lambdaBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
    cmdList->ResourceBarrier(1, &lambdaBarrier);
}

void ParticleSystem::DispatchInit(ID3D12GraphicsCommandList *cmdList, float dt)
{

    // upload constants
    struct NSConstants {
        XMFLOAT3 gridOrigin; float cellSize;
        int gridDimX, gridDimY, gridDimZ;
        int numParticles;
        int numCells;
        float dt;
        int _pad[2];
    };
    NSConstants cb;
    cb.gridOrigin = XMFLOAT3(-3.5f, 0.0f, -3.5f);
    cb.cellSize = 1.0f;  // smoothing radius
    cb.gridDimX = NS_GRID_DIM_X;
    cb.gridDimY = NS_GRID_DIM_Y;
    cb.gridDimZ = NS_GRID_DIM_Z;
    cb.numParticles = (int)NUM_PARTICLES;
    cb.numCells = NS_NUM_CELLS;
    cb.dt = dt;

    void* mapped = nullptr;
    m_nsConstantBuffer->Map(0, nullptr, &mapped);
    memcpy(mapped, &cb, sizeof(cb));
    m_nsConstantBuffer->Unmap(0, nullptr);

    // set root signature
    cmdList->SetComputeRootSignature(m_computeRootSignature.Get());
    cmdList->SetComputeRootConstantBufferView(0, m_nsConstantBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(1, m_nsCellCount->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(2, m_nsIntraOffset->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(3, m_nsParticlesIn->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, m_nsCellStart->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, m_nsStatusBuf->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(6, m_nsParticlesOut->GetGPUVirtualAddress());
}

void ParticleSystem::DispatchPrediction(ID3D12GraphicsCommandList *cmdList, float dt) {

    // upload particle positions to the gpu
    std::vector<GPUParticle> gpu(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        gpu[i].position = m_particles[i].position;
        gpu[i].predictedPosition = m_particles[i].predictedPosition;
        gpu[i].velocity = m_particles[i].velocity;
        gpu[i].density = m_particles[i].density;
        gpu[i].lambda = m_particles[i].lambda;

        std::cout << gpu[i].lambda << std::endl;
    }
    void* mapped = nullptr;
    m_nsUploadBuffer->Map(0, nullptr, &mapped);
    memcpy(mapped, gpu.data(), NUM_PARTICLES * sizeof(GPUParticle));
    m_nsUploadBuffer->Unmap(0, nullptr);

    // copy resource to gpu
    auto toDst = CD3DX12_RESOURCE_BARRIER::Transition(
        m_nsParticlesIn.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &toDst);
    cmdList->CopyResource(m_nsParticlesIn.Get(), m_nsUploadBuffer.Get());
    auto toUAV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_nsParticlesIn.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->ResourceBarrier(1, &toUAV);

    // --- 3. Dispatch the prediction kernel ---
    if (dt > 0.0f) {
        cmdList->SetPipelineState(m_psoPrediction.Get());
        cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);

        // barrier
        auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
        cmdList->ResourceBarrier(1, &uavBarrier);
    }
}

void ParticleSystem::DispatchNeighborSearch(ID3D12GraphicsCommandList *cmdList)
{
    // clear pass 
    cmdList->SetPipelineState(m_psoClear.Get());
    cmdList->Dispatch((NS_NUM_CELLS + 63) / 64, 1, 1);

    {
        D3D12_RESOURCE_BARRIER clearBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_nsCellCount.Get());
        cmdList->ResourceBarrier(1, &clearBarrier);
    }

    // step 1: count pass
    cmdList->SetPipelineState(m_psoCount.Get());
    cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER countBarriers[2] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_nsCellCount.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_nsIntraOffset.Get()),
    };
    cmdList->ResourceBarrier(2, countBarriers);

    //step 2: prefix sum pass
    cmdList->SetPipelineState(m_psoClearStatus.Get());
    cmdList->Dispatch((NS_NUM_CELLS + 255) / 256, 1, 1);  

    {
        D3D12_RESOURCE_BARRIER clearBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_nsStatusBuf.Get());
        cmdList->ResourceBarrier(1, &clearBarrier);
    }

    cmdList->SetPipelineState(m_psoPrefixScanPass.Get());
    cmdList->Dispatch((NS_NUM_CELLS + 255) / 256, 1, 1);

    D3D12_RESOURCE_BARRIER scanBarrier =
        CD3DX12_RESOURCE_BARRIER::UAV(m_nsCellStart.Get());
    cmdList->ResourceBarrier(1, &scanBarrier);

    // reorder pass -- depends on cellStart and intraOffset both being ready
    cmdList->SetPipelineState(m_psoReorder.Get());
    cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER reorderBarrier =
        CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesOut.Get());
    cmdList->ResourceBarrier(1, &reorderBarrier);
}

void ParticleSystem::ValidatePrefixSum(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12CommandQueue* cmdQueue,
    ID3D12Fence* fence,
    UINT64& fenceValue,
    HANDLE fenceEvent)
{
    // copy both buffers to readback
    D3D12_RESOURCE_BARRIER toSrc[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_nsCellCount.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_nsCellStart.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    cmdList->ResourceBarrier(2, toSrc);

    cmdList->CopyResource(m_nsReadbackCellCount.Get(), m_nsCellCount.Get());
    cmdList->CopyResource(m_nsReadbackCellStart.Get(), m_nsCellStart.Get());

    D3D12_RESOURCE_BARRIER toUAV[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_nsCellCount.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_nsCellStart.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    cmdList->ResourceBarrier(2, toUAV);

    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, lists);
    fenceValue++;
    ThrowIfFailed(cmdQueue->Signal(fence, fenceValue));
    ThrowIfFailed(fence->SetEventOnCompletion(fenceValue, fenceEvent));
    WaitForSingleObjectEx(fenceEvent, INFINITE, FALSE);

    // map both
    int* counts = nullptr;
    int* starts = nullptr;
    CD3DX12_RANGE range(0, NS_NUM_CELLS * sizeof(int));
    m_nsReadbackCellCount->Map(0, &range, reinterpret_cast<void**>(&counts));
    m_nsReadbackCellStart->Map(0, &range, reinterpret_cast<void**>(&starts));

    char buf[128];

    // test 1: cellStart[0] == 0
    sprintf_s(buf, "[PREFIX] cellStart[0] = %d (expected 0)\n", starts[0]);
    OutputDebugStringA(buf);

    // test 2: cellStart[i] + cellCount[i] == cellStart[i+1] for every i
    bool prefixOK = true;
    int firstBadCell = -1;
    for (int i = 0; i < NS_NUM_CELLS - 1; i++) {
        if (starts[i] + counts[i] != starts[i + 1]) {
            prefixOK = false;
            firstBadCell = i;
            break;
        }
    }

    if (prefixOK) {
        OutputDebugStringA("[PREFIX] prefix sum OK\n");
    } else {
        sprintf_s(buf, "[PREFIX] BROKEN at cell %d: start=%d count=%d sum=%d but next start=%d\n",
            firstBadCell,
            starts[firstBadCell],
            counts[firstBadCell],
            starts[firstBadCell] + counts[firstBadCell],
            starts[firstBadCell + 1]);
        OutputDebugStringA(buf);
    }

    // test 3: last cell's start + count == NUM_PARTICLES
    int last = NS_NUM_CELLS - 1;
    sprintf_s(buf, "[PREFIX] cellStart[last]+count = %d (expected %d)\n",
        starts[last] + counts[last], (int)NUM_PARTICLES);
    OutputDebugStringA(buf);

    // print the non-empty cells so you can eyeball the starts
    OutputDebugStringA("[PREFIX] non-empty cells:\n");
    int printed = 0;
    for (int i = 0; i < NS_NUM_CELLS && printed < 16; i++) {
        if (counts[i] > 0) {
            sprintf_s(buf, "[PREFIX]   cell[%d]: start=%d count=%d end=%d\n",
                i, starts[i], counts[i], starts[i] + counts[i]);
            OutputDebugStringA(buf);
            printed++;
        }
    }

    m_nsReadbackCellCount->Unmap(0, nullptr);
    m_nsReadbackCellStart->Unmap(0, nullptr);
}

// --------- ALL THE ACTUAL MATH STUFF GOES DOWN HERE -----------

XMFLOAT3 SubtractFloat3(XMFLOAT3 v1, XMFLOAT3 v2) {
    return XMFLOAT3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

float SquaredNorm(XMFLOAT3 vec) {
    float squared = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    return squared;
}

float Poly6(XMFLOAT3 r, float h) {
    const float coeff = 315.0 / (64.0 * XM_PI);

    float h2 = h * h;
    float r2 = SquaredNorm(r);

    if (r2 > h2) {
        return 0.0;
    }

    float h9 = pow(h, 9);
    float diff = h2 - r2;
    float diff3 = diff * diff * diff;

    return (coeff / h9) * diff3;
}


XMFLOAT3 SpikyGradient(XMFLOAT3 r, float h) {
    float coeff = 45.0f / XM_PI;

    float norm = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);

    if (norm > h) {
        return XMFLOAT3(0.0, 0.0, 0.0);
    }

    float h6 = pow(h, 6);
    float diff = h - norm;
    float diff2 = diff * diff;

    float times = (coeff / (h6 * max(norm, 1e-24f))) * diff2;
    XMFLOAT3 result = XMFLOAT3(-r.x * times, -r.y * times, -r.z * times);
    return result;
}

float ParticleSystem::ComputeDensityConstraint(int i, float radius) {
    Particle p_target = m_particles[i];

    float density = 0.0;
    for (int j : p_target.neighbors) {
        Particle p = m_particles[j];

        XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
        density += Poly6(pos, radius);
    }

    return density;
}

XMFLOAT3 ParticleSystem::ComputeGradiantConstraint(int i, int j, float density, float radius) {
    Particle p_target = m_particles[i];

    if (i == j) {
        XMFLOAT3 sum = XMFLOAT3(0.0, 0.0, 0.0);
        for (int neighbor : m_particles[i].neighbors) {
            Particle p = m_particles[neighbor];

            XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
            XMFLOAT3 grad = SpikyGradient(pos, radius);
            sum.x += grad.x;
            sum.y += grad.y;
            sum.z += grad.z;
        }

        XMFLOAT3 result = XMFLOAT3(sum.x / density, sum.y / density, sum.z / density);
        return result;
    } else {
        Particle p = m_particles[j];

        XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
        XMFLOAT3 grad = SpikyGradient(pos, radius);
        return XMFLOAT3(-grad.x, -grad.y, -grad.z);
    }
}

void ParticleSystem::CopyBackResources(ID3D12GraphicsCommandList* cmdList) {
    D3D12_RESOURCE_BARRIER toSrc[4] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesOut.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsCellCount.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsCellStart.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesIn.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    cmdList->ResourceBarrier(4, toSrc);

    cmdList->CopyResource(m_nsReadbackParticlesOut.Get(), m_nsParticlesOut.Get());
    cmdList->CopyResource(m_nsReadbackCellCount.Get(),    m_nsCellCount.Get());
    cmdList->CopyResource(m_nsReadbackCellStart.Get(),    m_nsCellStart.Get());
    cmdList->CopyResource(m_nsReadbackParticlesIn.Get(),  m_nsParticlesIn.Get());

    D3D12_RESOURCE_BARRIER toUAV[4] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesOut.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsCellCount.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsCellStart.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesIn.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    cmdList->ResourceBarrier(4, toUAV);
}

void ParticleSystem::ReadbackParticleData(ID3D12GraphicsCommandList* cmdList)
{
    GPUParticle* readback = nullptr;
    CD3DX12_RANGE readRange(0, NUM_PARTICLES * sizeof(GPUParticle));
    m_nsReadbackParticlesIn->Map(0, &readRange,
        reinterpret_cast<void**>(&readback));

    for (int i = 0; i < NUM_PARTICLES; i++) {
        m_particles[readback[i].originalIndex].predictedPosition = readback[i].predictedPosition;
        m_particles[readback[i].originalIndex].velocity = readback[i].velocity;
    }

    CD3DX12_RANGE writeRange(0, 0);
    m_nsReadbackParticlesIn->Unmap(0, &writeRange);
}

void ParticleSystem::UpdatePBD(float dt, ID3D12GraphicsCommandList* cmdList) {

    if (dt <= 0.0f) return;  // skip PBD on frame 1

	const float radius      = 1.0f;
    const float radius2     = radius * radius;
    const float rho_0       = 10.0f;       // rest density
    const float epsilon     = 100.0f;
    const float damping     = 0.999f;
    const float boxSize     = 3.0f;
    const float restitution = 0.3f;
    const float viscosity   = 0.05f;
    const int iterations    = 3;

	// calculations for all particles
    std::vector<XMFLOAT3> deltas(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});
    for (int iter = 0; iter < iterations; iter++) {

        // for all particles, calculate lambda_i
        for (int i = 0; i < NUM_PARTICLES; i++) {
            
            // ------------ STEP 3 : COMPUTE DENSITIES rho_i --------------
            float rho_i = ComputeDensityConstraint(i, radius);

            // ------------ STEP 4 : COMPUTE LAMBDA --------------
            float constraint = (rho_i / rho_0) - 1.0; // numerator
            float denominator = 0.0;
            for (int j : m_particles[i].neighbors) {
                XMFLOAT3 gradConstraint = ComputeGradiantConstraint(i, j, rho_0, radius);
                denominator += SquaredNorm(gradConstraint);
            }

            // add an epislon value for relaxation
            denominator += epsilon;
            m_particles[i].lambda = -constraint / denominator;
        }

        // ------------ STEP 5 : COMPUTE DENSITIES DELTA RHO --------------

        for (int i = 0; i < NUM_PARTICLES; i++) {
            Particle p = m_particles[i];

            // artifical tensile pressure correction constants
            const float corr_n = 4.0;
            const float corr_h = 0.30;
            const float corr_k = 1e-04;
            const float corr_w = Poly6(XMFLOAT3(corr_h * radius, 0.0, 0.0), radius);

            float sum_x = 0.0;
            float sum_y = 0.0;
            float sum_z = 0.0;

            for (int j : p.neighbors) {
                Particle p_neighbor = m_particles[j];

                // artificial tensile pressure correction 
                const float poly = Poly6(SubtractFloat3(p.predictedPosition, p_neighbor.predictedPosition), radius);
                const float ratio = poly / corr_w;
                const float corr_coeff = -corr_k * pow(ratio, corr_n);
                const float coeff = p.lambda + p_neighbor.lambda + corr_coeff;

                XMFLOAT3 sum = SpikyGradient(SubtractFloat3(p.predictedPosition, p_neighbor.predictedPosition), radius);
                sum_x += sum.x * coeff;
                sum_y += sum.y * coeff;
                sum_z += sum.z * coeff;
            }

            float rho_coeff = 1.0 / rho_0;
            XMFLOAT3 sum_vec = XMFLOAT3(rho_coeff * sum_x, rho_coeff * sum_y, rho_coeff * sum_z);

            // solve for delta rho
            deltas[i].x += sum_vec.x;
            deltas[i].y += sum_vec.y;
            deltas[i].z += sum_vec.z;
        }

        // ------------ STEP 6 : APPLY DELTA RHO --------------
		for (int i = 0; i < NUM_PARTICLES; i++) {
			m_particles[i].predictedPosition.x += deltas[i].x;
			m_particles[i].predictedPosition.y += deltas[i].y;
			m_particles[i].predictedPosition.z += deltas[i].z;
		}
    }

    std::vector<float> densities(NUM_PARTICLES, 0.0f);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        densities[i] = ComputeDensityConstraint(i, radius);
    }

    // XSPH viscosity computation
    std::vector<XMFLOAT3> xsph(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j : m_particles[i].neighbors) {
            float val = Poly6(SubtractFloat3(m_particles[i].position, m_particles[j].position), radius);
            XMFLOAT3 velocity = SubtractFloat3(m_particles[j].velocity, m_particles[i].velocity);

            float coeff = 1.0f / densities[i];
            xsph[i].x += coeff * val * velocity.x;
            xsph[i].y += coeff * val * velocity.y;
            xsph[i].z += coeff * val * velocity.z;
        }
    }

	// update positions and velocity
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMFLOAT3& pred = m_particles[i].predictedPosition;

        // constraints
        pred.x = max(pred.x, -boxSize); 
        pred.x = min(pred.x, boxSize);
        pred.y = max(pred.y, 0.0f); 
        pred.y = min(pred.y, 8.0f);
        pred.z = max(pred.z, -boxSize); 
        pred.z = min(pred.z, boxSize);

        // in the finalize loop, after clamping
        if (pred.y <= 0.0f) m_particles[i].velocity.y = 0.0f;
        if (pred.y >= 8.0f) m_particles[i].velocity.y = 0.0f;
        if (abs(pred.x) >= boxSize) m_particles[i].velocity.x = 0.0f;
        if (abs(pred.z) >= boxSize) m_particles[i].velocity.z = 0.0f;

        // update velocity
        m_particles[i].velocity.x = damping * (pred.x - m_particles[i].position.x) / dt;
		m_particles[i].velocity.y = damping * (pred.y - m_particles[i].position.y) / dt;
		m_particles[i].velocity.z = damping * (pred.z - m_particles[i].position.z) / dt;

        // apply XSPH velocity
        m_particles[i].velocity.x += xsph[i].x * viscosity;
        m_particles[i].velocity.y += xsph[i].y * viscosity;
        m_particles[i].velocity.z += xsph[i].z * viscosity;

        // TODO: vorticity confinement

        // update position 
		m_particles[i].position = pred;
	}

	UpdateInstances();
}

void ParticleSystem::UpdateInstances() {
    // push to instances vector
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMMATRIX mat = XMMatrixTranslation(
            m_particles[i].position.x,
            m_particles[i].position.y,
            m_particles[i].position.z
        );

        XMStoreFloat4x4(&m_instancer.m_instances[i].worldMatrix, mat);
    }

    memcpy(m_instancer.m_pInstanceDataBegin, m_instancer.m_instances.data(), sizeof(InstanceData) * NUM_PARTICLES);
}
