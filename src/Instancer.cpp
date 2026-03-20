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
    //CreateComputePipeline(device, numParticles, computePath);
}

void Instancer::CreateInstanceBuffer(ID3D12Device *device, UINT numParticles)
{   
    const UINT instanceBufferSize = numParticles * sizeof(InstanceData);

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(instanceBufferSize);
    ThrowIfFailed(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_instanceBuffer)));

    CD3DX12_RANGE readRange(0, 0);
    ThrowIfFailed(m_instanceBuffer->Map(0, &readRange,
        reinterpret_cast<void**>(&m_pInstanceDataBegin)));

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
    CD3DX12_DESCRIPTOR_RANGE ranges[2];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsConstants(1, 0);           // b0: numParticles
    params[1].InitAsDescriptorTable(2, ranges); // t0 + u0

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(2, params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(
        &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(device->CreateRootSignature(
        0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&m_computeRootSignature)));

    // compile compute shader
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
