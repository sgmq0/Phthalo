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
    void CreateDescriptorHeap(ID3D12Device* device, UINT numParticles);

    // doing the simulation
    void ComputeInstances(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12CommandQueue* cmdQueue,
        const Particle* particles,
        UINT numParticles);
    
    void TransitionToVertexBuffer(ID3D12GraphicsCommandList* cmdList); // start using the particles (SRV)
    void TransitionToUAV(ID3D12GraphicsCommandList* cmdList); // start using the instances (UAV)

    // sphere to draw
    SphereMesh m_sphere;
    size_t m_sphereIndexCount;

    ComPtr<ID3D12Resource> m_instanceBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_instanceBufferView;
    // UINT8* m_pInstanceDataBegin = nullptr;

    std::vector<InstanceData> m_instances;
    // UINT m_numParticles;

    // buffers that help us push the stuff to the gpu
    ComPtr<ID3D12Resource> m_particleUploadBuffer; // where cpu writes
    ComPtr<ID3D12Resource> m_particleDefaultBuffer; // where gpu reads

    // things to setup the compute pipeline
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12PipelineState> m_computePipelineState;

    // things for the descriptor heap
    ComPtr<ID3D12DescriptorHeap> m_computeHeap;
    UINT m_computeDescriptorSize;

    bool m_instanceBufferInitialized = false;

private:

};
