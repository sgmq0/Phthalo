#pragma once
#include "stdafx.h"

#include "SphereMesh.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

class Instancer {
public:
    Instancer();

    // called once in D3D12Renderer::LoadAssets(), basically just sets up everything here.
    void Init(ID3D12Device* device, UINT numParticles, std::wstring computePath);

    // different setup functions
    void CreateInstanceBuffer(ID3D12Device* device, UINT numParticles);
    void CreateParticleBuffers(ID3D12Device* device, UINT numParticles);
    void CreateComputePipeline(ID3D12Device* device, UINT numParticles, std::wstring computePath);

    // sphere to draw
    SphereMesh m_sphere;
    size_t m_sphereIndexCount;

    ComPtr<ID3D12Resource> m_instanceBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_instanceBufferView;
    UINT8* m_pInstanceDataBegin = nullptr;

    std::vector<InstanceData> m_instances;
    // UINT m_numParticles;

    // buffers that help us push the stuff to the gpu
    ComPtr<ID3D12Resource> m_particleUploadBuffer; // where cpu writes
    ComPtr<ID3D12Resource> m_particleDefaultBuffer; // where gpu reads

    // things to setup the compute pipeline
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12PipelineState> m_computePipelineState;

private:

};
