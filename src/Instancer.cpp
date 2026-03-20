#include "stdafx.h"
#include "Instancer.h"
#include "DXSampleHelper.h"

Instancer::Instancer() :
    m_sphere(SphereMesh(0.1f))
{
    m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();
}

void Instancer::Init(ID3D12Device* device, UINT numParticles)
{
    CreateInstanceBuffer(device, numParticles);
    CreateParticleBuffers(device, numParticles);
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
        IID_PPV_ARGS(&m_instanceBuffer)));  // no longer m_particleSystem.m_instancer.

    CD3DX12_RANGE readRange(0, 0);
    ThrowIfFailed(m_instanceBuffer->Map(0, &readRange,
        reinterpret_cast<void**>(&m_pInstanceDataBegin)));

    m_instanceBufferView.BufferLocation = m_instanceBuffer->GetGPUVirtualAddress();
    m_instanceBufferView.StrideInBytes  = sizeof(InstanceData);
    m_instanceBufferView.SizeInBytes    = instanceBufferSize;
}

void Instancer::CreateParticleBuffers(ID3D12Device *device, UINT numParticles)
{
}
