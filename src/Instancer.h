#pragma once
#include "stdafx.h"

#include "SphereMesh.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

class Instancer {
public:
    Instancer();

    // called once in D3D12Renderer::LoadAssets(), basically just sets up everything here.
    void Init(ID3D12Device* device, UINT numParticles);

    // different setup functions
    void CreateInstanceBuffer(ID3D12Device* device, UINT numParticles);
    void CreateParticleBuffers(ID3D12Device* device, UINT numParticles);

    SphereMesh m_sphere;
    size_t m_sphereIndexCount;

    ComPtr<ID3D12Resource> m_instanceBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_instanceBufferView;
    UINT8* m_pInstanceDataBegin = nullptr;

    std::vector<InstanceData> m_instances;
    // UINT m_numParticles;

private:

};
