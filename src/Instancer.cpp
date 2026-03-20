#include "stdafx.h"
#include "Instancer.h"
#include "DXSampleHelper.h"
#include "DXApplication.h"

Instancer::Instancer() :
    m_sphere(SphereMesh(0.1f))
{
    m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();
}

void Instancer::Init(ID3D12Device* device, UINT numParticles, std::wstring computePath)
{
    CreateInstanceBuffer(device, numParticles);
    CreateParticleBuffers(device, numParticles);
    CreateComputePipeline(device, numParticles, computePath);
    CreateDescriptorHeap(device, numParticles);
}

void Instancer::CreateInstanceBuffer(ID3D12Device *device, UINT numParticles)
{   
    const UINT instanceBufferSize = numParticles * sizeof(InstanceData);

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(instanceBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_instanceBuffer)));

    m_instanceBufferView.BufferLocation = m_instanceBuffer->GetGPUVirtualAddress();
    m_instanceBufferView.StrideInBytes = sizeof(InstanceData);
    m_instanceBufferView.SizeInBytes = instanceBufferSize;
}

void Instancer::CreateParticleBuffers(ID3D12Device *device, UINT numParticles)
{
    // step 1: Create upload heap
    // this is what the CPU writes to (in ParticleSystem::UpdateInstances()?)
    {
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(numParticles * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_particleUploadBuffer)));
    }

    // step 2: Create default heap
    // this is what the GPU reads from 
    {
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(numParticles * sizeof(GPUParticle));
        ThrowIfFailed(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_particleDefaultBuffer)));
    }
}

void Instancer::CreateComputePipeline(ID3D12Device* device, UINT numParticles, std::wstring computePath)
{
    CD3DX12_ROOT_PARAMETER params[3];
    params[0].InitAsConstants(1, 0); // b0: numParticles
    params[1].InitAsShaderResourceView(0); // t0: particlesIn
    params[2].InitAsUnorderedAccessView(0); // u0: instancesOut;

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(3, params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(
        &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(device->CreateRootSignature(
        0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_computeRootSignature)));

    //compile compute shader
    ComPtr<ID3DBlob> csBlob, csErr;
    HRESULT hr = D3DCompileFromFile(
        computePath.c_str(), nullptr, nullptr,
        "CSUpdateInstances", "cs_5_0",
        0, 0, &csBlob, &csErr);
    if (FAILED(hr)) {
        if (csErr) OutputDebugStringA((char*)csErr->GetBufferPointer());
        ThrowIfFailed(hr);
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_computeRootSignature.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(csBlob.Get());
    ThrowIfFailed(device->CreateComputePipelineState(
        &psoDesc, IID_PPV_ARGS(&m_computePipelineState)));
}

void Instancer::CreateDescriptorHeap(ID3D12Device *device, UINT numParticles)
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 2;
    // two slots, one for the SRV (read only buffer, particlesIn), 
    // one for the UAV (read and write, instancesOut)
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_computeHeap)));

    m_computeDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // SRV (t0) for particlesIn (readonly)
    D3D12_SHADER_RESOURCE_VIEW_DESC particlesInDesc = {};
    particlesInDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    particlesInDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    particlesInDesc.Buffer.NumElements = numParticles;
    particlesInDesc.Buffer.StructureByteStride = sizeof(GPUParticle);
    particlesInDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_computeHeap->GetCPUDescriptorHandleForHeapStart(), 0, m_computeDescriptorSize);
    device->CreateShaderResourceView(m_particleDefaultBuffer.Get(), &particlesInDesc, srvHandle);

    // UAV (u0) for instancesOut (readwrite)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.NumElements = numParticles;
    uavDesc.Buffer.StructureByteStride = sizeof(InstanceData);

    CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_computeHeap->GetCPUDescriptorHandleForHeapStart(), 1, m_computeDescriptorSize);
    device->CreateUnorderedAccessView(m_instanceBuffer.Get(), nullptr, &uavDesc, uavHandle);
}

void Instancer::ComputeInstances(ID3D12GraphicsCommandList *cmdList, ID3D12CommandQueue *cmdQueue, const Particle *particles, UINT numParticles)
{
    TransitionToUAV(cmdList); // start working with the instances

    // step 1: pack CPU particles into GPU-struct
    // ignoring the neighbors.... we'll figure this out soon...
    std::vector<GPUParticle> gpuParticles(numParticles);
    for (UINT i = 0; i < numParticles; i++) {
        gpuParticles[i].position = particles[i].position;
        gpuParticles[i].predictedPosition = particles[i].predictedPosition;
        gpuParticles[i].velocity = particles[i].velocity;
        gpuParticles[i].density = particles[i].density;
        gpuParticles[i].lambda = particles[i].lambda;
    }

    // step 2: CPU memcpy's into the upload buffer
    void* mapped = nullptr;
    m_particleUploadBuffer->Map(0, nullptr, &mapped);
    memcpy(mapped, gpuParticles.data(), numParticles * sizeof(GPUParticle));
    m_particleUploadBuffer->Unmap(0, nullptr);

    // step 3: copy resource from upload buffer (CPU) to default buffer (GPU)
    cmdList->CopyResource(m_particleDefaultBuffer.Get(), m_particleUploadBuffer.Get());

    // step 4: transition particle buffer COPY_DEST → SRV
    // copy is done, now compute shader can read it
    CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_particleDefaultBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &toSRV);

    // step 5: bind compute pipeline and resources
    cmdList->SetPipelineState(m_computePipelineState.Get());
    cmdList->SetComputeRootSignature(m_computeRootSignature.Get());
    cmdList->SetComputeRoot32BitConstants(0, 1, &numParticles, 0);
    cmdList->SetComputeRootShaderResourceView(1, m_particleDefaultBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(2, m_instanceBuffer->GetGPUVirtualAddress());

    // step 6: dispatch into groups of 64
    UINT groups = (numParticles + 63) / 64;
    cmdList->Dispatch(groups, 1, 1);

    CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_instanceBuffer.Get());
    cmdList->ResourceBarrier(1, &uavBarrier);

    // step 7: transition particle buffer back SRV → COPY_DEST
    // ready for next frame's upload copy
    CD3DX12_RESOURCE_BARRIER toCopyDest = CD3DX12_RESOURCE_BARRIER::Transition(
        m_particleDefaultBuffer.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &toCopyDest);
}

void Instancer::TransitionToVertexBuffer(ID3D12GraphicsCommandList* cmdList)
{
     // swap from uav to srv
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_instanceBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    cmdList->ResourceBarrier(1, &barrier);
}

void Instancer::TransitionToUAV(ID3D12GraphicsCommandList* cmdList)
{
    // swap from srv to uav
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_instanceBuffer.Get(),
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->ResourceBarrier(1, &barrier);
}