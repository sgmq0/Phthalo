#include "stdafx.h"
#include "ParticleSystem.h"
#include <iostream>
#include <algorithm>

ParticleSystem::ParticleSystem() 
{
    m_instancer = Instancer(PARTICLE_SIZE);
}

void ParticleSystem::LoadParticles()
{
    m_particles.resize(NUM_PARTICLES);

	// initializes all the particles
    int perAxis = (int)cbrt(NUM_PARTICLES);
    int i = 0;
    for (int x = 0; x < perAxis; x++)
    for (int y = 0; y < perAxis; y++)
    for (int z = 0; z < perAxis; z++) {
        if (i >= NUM_PARTICLES) break;

        m_particles[i].position = {
            (x - perAxis/2.0f) * PARTICLE_SPACING,
            y * PARTICLE_SPACING + 1.0f,   // start above floor
            (z - perAxis/2.0f) * PARTICLE_SPACING
        };
        m_particles[i].velocity = {0, 0, 0};
        m_particles[i].predictedPosition = m_particles[i].position; 
        m_particles[i].density = 0.0f;
        m_particles[i].lambda = 0.0f;
        m_particles[i].xsph = {0.0, 0.0, 0.0};
        i++;
    }
    
    m_instancer.m_instances.resize(NUM_PARTICLES);
}

ComPtr<ID3DBlob> CompileHelper(std::wstring shaderPath, const char* entry) 
{
    ComPtr<ID3DBlob> computeShader, computeError;
    HRESULT hr = D3DCompileFromFile(shaderPath.c_str(),
        nullptr, nullptr, entry, "cs_5_0", 0, 0, &computeShader, &computeError);
    if (computeError) OutputDebugStringA((char*)computeError->GetBufferPointer());
    ThrowIfFailed(hr);
    return computeShader;
};

ComPtr<ID3D12PipelineState> MakePSOHelper(ComPtr<ID3DBlob> computeShader, ComPtr<ID3D12RootSignature> rootSignature, ID3D12Device* device) 
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

    ComPtr<ID3D12PipelineState> pso;
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));
    return pso;
};

ComPtr<ID3D12Resource> MakeBufferHelper(UINT byteSize, ID3D12Device* device) 
{
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

    CD3DX12_ROOT_PARAMETER params[11];
    params[0].InitAsConstantBufferView(0);  // b0: NSConstants
    params[1].InitAsConstantBufferView(1);  // b1: MCConstants

    // params for pbf
    params[2].InitAsUnorderedAccessView(0); // u0: cellCount
    params[3].InitAsUnorderedAccessView(1); // u1: intraOffset
    params[4].InitAsUnorderedAccessView(2); // u2: particlesIn
    params[5].InitAsUnorderedAccessView(3); // u3: cellStart
    params[6].InitAsUnorderedAccessView(4); // u4: groupSums
    params[7].InitAsUnorderedAccessView(5); // u5: particlesOut

    // params for marching cubes
    params[8].InitAsUnorderedAccessView(6); // u6: mcScalarField
    params[9].InitAsUnorderedAccessView(7); // u7: mcVertexBuffer
    params[10].InitAsUnorderedAccessView(8); // u8: mcArgs
    // params[10].InitAsUnorderedAccessView(9); // u9: mcVertexCounter

    CD3DX12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = 11;
    rootDesc.pParameters = params;
    rootDesc.NumStaticSamplers = 0;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(),
        sig->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));

    // 2/3. compile shaders & pipeline states

    // ----- uniform grid search kernels -----
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

    // ----- pbf kernels -----
    ComPtr<ID3DBlob> prediction = CompileHelper(shaderPath, "CSPrediction");
    ComPtr<ID3DBlob> computeLambda = CompileHelper(shaderPath, "CSComputeLambda");
    ComPtr<ID3DBlob> computeDelta = CompileHelper(shaderPath, "CSComputeDelta");
    ComPtr<ID3DBlob> collisionConstraints = CompileHelper(shaderPath, "CSCollisionConstraints");
    ComPtr<ID3DBlob> updateVelocity = CompileHelper(shaderPath, "CSUpdateVelocity");
    ComPtr<ID3DBlob> computeXSPH = CompileHelper(shaderPath, "CSComputeXSPH");
    ComPtr<ID3DBlob> finalize = CompileHelper(shaderPath, "CSFinalize");

    m_psoPrediction = MakePSOHelper(prediction.Get(), m_computeRootSignature.Get(), device);
    m_psoComputeLambda = MakePSOHelper(computeLambda.Get(), m_computeRootSignature.Get(), device);
    m_psoComputeDelta = MakePSOHelper(computeDelta.Get(), m_computeRootSignature.Get(), device);
    m_psoCollisionConstraints = MakePSOHelper(collisionConstraints.Get(), m_computeRootSignature.Get(), device);
    m_psoUpdateVelocity = MakePSOHelper(updateVelocity.Get(), m_computeRootSignature.Get(), device);
    m_psoComputeXSPH = MakePSOHelper(computeXSPH.Get(), m_computeRootSignature.Get(), device);
    m_psoComputeFinalize = MakePSOHelper(finalize.Get(), m_computeRootSignature.Get(), device);

    // ----- marching cubes kernels ----- 
    ComPtr<ID3DBlob> clearArgs = CompileHelper(shaderPath, "CSClearArgs");
    ComPtr<ID3DBlob> clearField = CompileHelper(shaderPath, "CSClearField");
    ComPtr<ID3DBlob> buildScalarField = CompileHelper(shaderPath, "CSBuildScalarField");
    ComPtr<ID3DBlob> marchingCubes = CompileHelper(shaderPath, "CSMarchingCubes");
    m_psoClearArgs = MakePSOHelper(clearArgs.Get(), m_computeRootSignature.Get(), device);
    m_psoClearField = MakePSOHelper(clearField.Get(), m_computeRootSignature.Get(), device);
    m_psoBuildField = MakePSOHelper(buildScalarField.Get(), m_computeRootSignature.Get(), device);
    m_psoMarchingCubes = MakePSOHelper(marchingCubes.Get(), m_computeRootSignature.Get(), device);

    // 4. all of the buffers
    m_nsParticlesIn = MakeBufferHelper(NUM_PARTICLES * sizeof(GPUParticle), device);
    m_nsCellCount = MakeBufferHelper(NS_NUM_CELLS * sizeof(int), device);
    m_nsIntraOffset = MakeBufferHelper(NUM_PARTICLES * sizeof(int), device);
    m_nsCellStart = MakeBufferHelper(NS_NUM_CELLS * sizeof(int), device);
    UINT numGroups = (NS_NUM_CELLS + 255) / 256;
    m_nsStatusBuf = MakeBufferHelper(numGroups * sizeof(int), device);
    m_nsParticlesOut = MakeBufferHelper(NUM_PARTICLES * sizeof(GPUParticle), device);

    m_mcScalarField = MakeBufferHelper((MC_DIM_X+1) * (MC_DIM_Y+1) * (MC_DIM_Z+1) * sizeof(float), device);
    m_mcVertexBuffer = MakeBufferHelper(MC_MAX_TRIS * 3 * sizeof(XMFLOAT4), device);
    m_mcIndirectArgs = MakeBufferHelper(sizeof(int), device);   // TODO: change this later
    // m_mcVertexCounter;

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

    // const buffers (256-byte aligned)
    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        UINT sz = (sizeof(NSConstants) + 255) & ~255;
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_nsConstantBuffer)));
    }

    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        UINT sz = (sizeof(MCConstants) + 255) & ~255;
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_mcConstantBuffer)));
    }

    // readback buffers
    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(NUM_PARTICLES * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_nsReadbackParticlesIn))); // new ComPtr member
    }

    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(MC_MAX_TRIS * 3 * sizeof(XMFLOAT4));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_mcReadbackVertexBuffer))); // new ComPtr member
    }

    {
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(int));
        ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_mcReadbackArgs))); // new ComPtr member
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
    DispatchPrediction(cmdList, dt); // step 1: predict position
    DispatchNeighborSearch(cmdList); // step 2: perform neighbor search

    for (int i = 0; i < ITERATIONS; i++) {
        
        // step 3: 
        // compute lambda_i
        cmdList->SetPipelineState(m_psoComputeLambda.Get());
        cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
        auto lambdaBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
        cmdList->ResourceBarrier(1, &lambdaBarrier);

        // compute delta rho
        // step 5: compute delta
        cmdList->SetPipelineState(m_psoComputeDelta.Get());
        cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
        auto b2 = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
        cmdList->ResourceBarrier(1, &b2);

        // perform collision detection and response
        // also updates predicted position using delta
        cmdList->SetPipelineState(m_psoCollisionConstraints.Get());
        cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
        auto b4 = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
        cmdList->ResourceBarrier(1, &b4);
    }

    // insert kernel to update velocity
    // cmdList->SetPipelineState(m_psoUpdateVelocity.Get());
    // cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
    // auto b5 = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
    // cmdList->ResourceBarrier(1, &b5);

    // xsph
    cmdList->SetPipelineState(m_psoComputeXSPH.Get());
    cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
    auto b3 = CD3DX12_RESOURCE_BARRIER::UAV(m_nsParticlesIn.Get());
    cmdList->ResourceBarrier(1, &b3);

    // insert kernel to update position
}

void ParticleSystem::DispatchMarchingCubes(ID3D12GraphicsCommandList *cmdList)
{

    // 0: clear args
    cmdList->SetPipelineState(m_psoClearArgs.Get());
    cmdList->Dispatch(1, 1, 1);
    auto b0 = CD3DX12_RESOURCE_BARRIER::UAV(m_mcIndirectArgs.Get());
    cmdList->ResourceBarrier(1, &b0);

    // 1: clear scalar field
    UINT fieldVerts = (MC_DIM_X+1)*(MC_DIM_Y+1)*(MC_DIM_Z+1);
    cmdList->SetPipelineState(m_psoClearField.Get());
    cmdList->Dispatch((fieldVerts + 63) / 64, 1, 1);
    auto b1 = CD3DX12_RESOURCE_BARRIER::UAV(m_mcScalarField.Get());
    cmdList->ResourceBarrier(1, &b1);

    // 2. splat particles
    cmdList->SetPipelineState(m_psoBuildField.Get());
    cmdList->Dispatch((NUM_PARTICLES + 63) / 64, 1, 1);
    auto b2 = CD3DX12_RESOURCE_BARRIER::UAV(m_mcScalarField.Get());
    cmdList->ResourceBarrier(1, &b2);

    // 3. perform marching cubes
    UINT numCells = MC_DIM_X * MC_DIM_Y * MC_DIM_Z;
    cmdList->SetPipelineState(m_psoMarchingCubes.Get());
    cmdList->Dispatch((numCells + 63) / 64, 1, 1);
    auto b3 = CD3DX12_RESOURCE_BARRIER::UAV(m_mcVertexBuffer.Get());
    cmdList->ResourceBarrier(1, &b3);

}

void ParticleSystem::DispatchInit(ID3D12GraphicsCommandList *cmdList, float dt)
{

    // upload constants
    struct NSConstants {
        XMFLOAT3 gridOrigin; 
        float cellSize;
        int gridDimX, gridDimY, gridDimZ;
        int numParticles;
        float sizeX, sizeY, sizeZ;
        float _pad;
        int numCells;
        float dt;
        float H;
        float rho_0;        // rest density
        float epsilon;
        float _pad1[3];
    };
    NSConstants cb;
    cb.gridOrigin = XMFLOAT3(-BBOX_SIZE_XZ - CELL_SIZE, - CELL_SIZE, -BBOX_SIZE_XZ - CELL_SIZE);
    cb.cellSize = CELL_SIZE;
    cb.gridDimX = NS_DIM_X;
    cb.gridDimY = NS_DIM_Y;
    cb.gridDimZ = NS_DIM_Z;
    cb.numParticles = (int)NUM_PARTICLES;
    cb.sizeX = BBOX_SIZE_XZ;
    cb.sizeY = BBOX_SIZE_Y;
    cb.sizeZ = BBOX_SIZE_XZ;
    cb.numCells = NS_NUM_CELLS;
    cb.dt = dt;
    cb.H = CELL_SIZE;   // cell size is also the smoothing radius
    cb.rho_0 = RHO_0;
    cb.epsilon = EPSILON;

    void* ns_mapped = nullptr;
    m_nsConstantBuffer->Map(0, nullptr, &ns_mapped);
    memcpy(ns_mapped, &cb, sizeof(cb));
    m_nsConstantBuffer->Unmap(0, nullptr);

    struct MCConstants {
        XMFLOAT3 mcOrigin; 
        float mcCellSize;
        XMINT3 mcDims;
        float mcIso;
        int mcMaxTris;
        float _pad[3];
    };
    MCConstants mc_cb;
    mc_cb.mcOrigin = XMFLOAT3(-BBOX_SIZE_XZ - MC_CELL_SIZE, - CELL_SIZE, -BBOX_SIZE_XZ - MC_CELL_SIZE);
    mc_cb.mcCellSize = MC_CELL_SIZE;
    mc_cb.mcDims = XMINT3(MC_DIM_X, MC_DIM_Y, MC_DIM_Z);
    mc_cb.mcIso = MC_ISO;
    mc_cb.mcMaxTris = MC_MAX_TRIS;

    void* mc_mapped = nullptr;
    m_mcConstantBuffer->Map(0, nullptr, &mc_mapped);
    memcpy(mc_mapped, &mc_cb, sizeof(mc_cb));
    m_mcConstantBuffer->Unmap(0, nullptr);

    // set root signature
    cmdList->SetComputeRootSignature(m_computeRootSignature.Get());
    cmdList->SetComputeRootConstantBufferView(0, m_nsConstantBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootConstantBufferView(1, m_mcConstantBuffer->GetGPUVirtualAddress());

    cmdList->SetComputeRootUnorderedAccessView(2, m_nsCellCount->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(3, m_nsIntraOffset->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, m_nsParticlesIn->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, m_nsCellStart->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(6, m_nsStatusBuf->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(7, m_nsParticlesOut->GetGPUVirtualAddress());

    // params for marching cubes
    cmdList->SetComputeRootUnorderedAccessView(8, m_mcScalarField->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(9, m_mcVertexBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(10, m_mcIndirectArgs->GetGPUVirtualAddress());
    // cmdList->SetComputeRootUnorderedAccessView(10, m_mcVertexCounter->GetGPUVirtualAddress());
}

void ParticleSystem::DispatchPrediction(ID3D12GraphicsCommandList *cmdList, float dt) 
{
    // upload particle positions to the gpu
    std::vector<GPUParticle> gpu(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        gpu[i].position = m_particles[i].position;
        gpu[i].predictedPosition = m_particles[i].predictedPosition;
        gpu[i].velocity = m_particles[i].velocity;
        gpu[i].density = m_particles[i].density;
        gpu[i].lambda = m_particles[i].lambda;
        gpu[i].xsph = m_particles[i].xsph;
        gpu[i].delta = m_particles[i].delta;
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

ComPtr<ID3D12PipelineState> ParticleSystem::GetPsoClear()
{
    return m_psoClear;
}

void ParticleSystem::CopyBackResources(ID3D12GraphicsCommandList* cmdList) 
{
    {
        D3D12_RESOURCE_BARRIER toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesIn.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toSrc);
        cmdList->CopyResource(m_nsReadbackParticlesIn.Get(),  m_nsParticlesIn.Get());
        D3D12_RESOURCE_BARRIER toUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_nsParticlesIn.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &toUAV);
    }

    {
        D3D12_RESOURCE_BARRIER toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_mcVertexBuffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toSrc);
        cmdList->CopyResource(m_mcReadbackVertexBuffer.Get(),  m_mcVertexBuffer.Get());
        D3D12_RESOURCE_BARRIER toUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_mcVertexBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &toUAV);
    }

    {
        D3D12_RESOURCE_BARRIER toSrc = CD3DX12_RESOURCE_BARRIER::Transition(m_mcIndirectArgs.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toSrc);
        cmdList->CopyResource(m_mcReadbackArgs.Get(),  m_mcIndirectArgs.Get());
        D3D12_RESOURCE_BARRIER toUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_mcIndirectArgs.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &toUAV);
    }
}

void ParticleSystem::ReadbackParticleData(ID3D12GraphicsCommandList* cmdList)
{
    GPUParticle* readback = nullptr;
    CD3DX12_RANGE readRange(0, NUM_PARTICLES * sizeof(GPUParticle));
    m_nsReadbackParticlesIn->Map(0, &readRange,
        reinterpret_cast<void**>(&readback));

    for (int i = 0; i < NUM_PARTICLES; i++) {
        m_particles[i].position = readback[i].position;
        m_particles[i].predictedPosition = readback[i].predictedPosition;
        m_particles[i].velocity = readback[i].velocity;
        m_particles[i].lambda = readback[i].lambda;
        m_particles[i].xsph = readback[i].xsph;
    }

    CD3DX12_RANGE writeRange(0, 0);
    m_nsReadbackParticlesIn->Unmap(0, &writeRange);
}

void ParticleSystem::ReadbackVertexData(ID3D12GraphicsCommandList *cmdList)
{
    uint32_t* args = nullptr;
    m_mcReadbackArgs->Map(0, nullptr, (void**)&args);
    uint32_t vertexCount = args[0];
    m_mcReadbackArgs->Unmap(0, nullptr);

    printf("Vertex count: %u\n", vertexCount);

    if (m_vertices.size() < vertexCount)
        m_vertices.resize(vertexCount);

    Vertex* readback = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    m_mcReadbackVertexBuffer->Map(0, &readRange,
        reinterpret_cast<void**>(&readback));

    for (int i = 0; i < vertexCount; i++) {
        m_vertices[i] = readback[i];
    }

    CD3DX12_RANGE writeRange(0, 0);
    m_mcReadbackVertexBuffer->Unmap(0, &writeRange);
}


// --------- ALL THE ACTUAL MATH STUFF GOES DOWN HERE -----------

void ParticleSystem::UpdatePBD(float dt, ID3D12GraphicsCommandList* cmdList) 
{
    if (dt <= 0.0f) return;  // skip PBD on frame 1

	// update positions and velocity
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMFLOAT3& pred = m_particles[i].predictedPosition;
        XMFLOAT3& pos  = m_particles[i].position;

        // derive velocity from the displacement (this is PBD -- velocity comes last)
        m_particles[i].velocity.x = DAMPING * (pred.x - pos.x) / dt;
        m_particles[i].velocity.y = DAMPING * (pred.y - pos.y) / dt;
        m_particles[i].velocity.z = DAMPING * (pred.z - pos.z) / dt;

        // todo: vorticity

        // apply XSPH (fix: use per-axis components)
        m_particles[i].velocity.x += m_particles[i].xsph.x * VISCOSITY;
        m_particles[i].velocity.y += m_particles[i].xsph.y * VISCOSITY;
        m_particles[i].velocity.z += m_particles[i].xsph.z * VISCOSITY;

        pos = pred;
    }

	UpdateInstances();
}

void ParticleSystem::UpdateInstances() 
{
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
