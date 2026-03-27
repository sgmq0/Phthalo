#pragma once
#include "stdafx.h"

#include "Instancer.h"
#include "DXApplication.h"

using namespace DirectX;

class ParticleSystem {
public:
    ParticleSystem();
    ParticleSystem(UINT numParticles);

    void LoadParticles();
    void Update(float dt, ID3D12GraphicsCommandList* cmdList);
    void UpdateInstances();

    float ComputeDensityConstraint(int i, float radius);
    XMFLOAT3 ComputeGradiantConstraint(int i, int j, float density, float radius);

    static const UINT NUM_PARTICLES = 1000;
    Particle m_particles[NUM_PARTICLES];

    Instancer m_instancer;

    // ----- the actual compute pipeline -----
    void CreateComputePipeline(
        ID3D12Device* device,
        std::wstring shaderPath,
        ComPtr<ID3D12CommandAllocator>& commandAllocator,
        ComPtr<ID3D12GraphicsCommandList>& commandList
    );
    void DispatchNeighborSearch(ID3D12GraphicsCommandList* cmdList);

    static constexpr int NS_GRID_DIM_X = 20;  // (-3 to +3) / cellSize = 0.35 -> 18, round up
    static constexpr int NS_GRID_DIM_Y = 26;  // 0 to 9 / 0.35 -> 26
    static constexpr int NS_GRID_DIM_Z = 20;
    static constexpr int NS_NUM_CELLS  = NS_GRID_DIM_X * NS_GRID_DIM_Y * NS_GRID_DIM_Z;
    
    bool m_nsFirstFrame = true;

    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12Resource> m_nsUploadBuffer;    // CPU -> GPU staging
    ComPtr<ID3D12Resource> m_nsConstantBuffer;  // b0: constant buffer

    // first pass: counting kernel
    ComPtr<ID3D12PipelineState> m_psoClear;
    ComPtr<ID3D12PipelineState> m_psoCount;

    ComPtr<ID3D12Resource> m_nsCellCount;       // u0: int per cell, zeroed each frame
    ComPtr<ID3D12Resource> m_nsIntraOffset;     // u1: int per particle, slot within cell
    ComPtr<ID3D12Resource> m_nsParticlesIn;     // u2: GPUParticle upload target

    // second pass: global prefix sum
    ComPtr<ID3D12PipelineState> m_psoClearStatus;
    ComPtr<ID3D12PipelineState> m_psoPrefixScanPass;

    ComPtr<ID3D12Resource> m_nsCellStart;       // u3: prefix sum output
    ComPtr<ID3D12Resource> m_nsStatusBuf;       // u4: scratch inter-group aggregates

    // readback
    ComPtr<ID3D12Resource> m_nsReadbackCellCount;
    ComPtr<ID3D12Resource> m_nsReadbackCellStart;

    void ValidateCounting(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12CommandQueue* cmdQueue,
        ID3D12Fence* fence,
        UINT64& fenceValue,
        HANDLE fenceEvent);
    void ParticleSystem::ValidatePrefixSum(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12CommandQueue* cmdQueue,
        ID3D12Fence* fence,
        UINT64& fenceValue,
        HANDLE fenceEvent);

    // testing
    ComPtr<ID3D12Resource> m_computeReadbackBuffer;

private:
    
};
