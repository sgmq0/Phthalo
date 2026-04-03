#pragma once
#include "stdafx.h"

#include "Instancer.h"
#include "DXApplication.h"

using namespace DirectX;

class ParticleSystem {
public:
    ParticleSystem();

    void LoadParticles();
    void CreateComputePipeline(
        ID3D12Device* device,
        std::wstring shaderPath,
        ComPtr<ID3D12CommandAllocator>& commandAllocator,
        ComPtr<ID3D12GraphicsCommandList>& commandList
    );

    // update calls
    void ReadbackParticleData(ID3D12GraphicsCommandList* cmdList);  // load back the particles to CPU
    void CopyBackResources(ID3D12GraphicsCommandList* cmdList);
    void UpdatePBD(float dt, ID3D12GraphicsCommandList* cmdList);
    void UpdateInstances();

    // compute dispatches
    void DispatchGPUCommands(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchInit(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchPrediction(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchNeighborSearch(ID3D12GraphicsCommandList* cmdList);

    // various getters for private variables
    ComPtr<ID3D12PipelineState> GetPsoClear();

    // instancing member variables
    std::vector<Particle> m_particles;
    Instancer m_instancer;

    // --------- CONSTANTS --------
    static const UINT NUM_PARTICLES = 50000;
    const float PARTICLE_SIZE = 0.1f;
    const float PARTICLE_SPACING = 0.3f;    // how far the particles spawn from each other

    // simulation consts
    const float BBOX_SIZE_XZ = 10.0f;
    const float BBOX_SIZE_Y = 100.0f;
    const float CELL_SIZE = 1.0f;
    const float RHO_0 = 40.0f;          // rest density
    const float EPSILON = 100.0f;

    // consts we use in finalization step
    const float DAMPING = 0.999f;
    const float VISCOSITY = 0.1f;
    const int ITERATIONS = 2;

    // uniform grid search consts
    const UINT NS_GRID_DIM_X = (UINT)ceil((BBOX_SIZE_XZ * 2) / CELL_SIZE) + 2;
    const UINT NS_GRID_DIM_Y = (UINT)ceil(BBOX_SIZE_Y / CELL_SIZE) + 2;
    const UINT NS_GRID_DIM_Z = (UINT)ceil((BBOX_SIZE_XZ * 2) / CELL_SIZE) + 2;
    const UINT NS_NUM_CELLS = NS_GRID_DIM_X * NS_GRID_DIM_Y * NS_GRID_DIM_Z;
    
    bool m_nsFirstFrame = true;

private:

    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12Resource> m_nsUploadBuffer;    // CPU -> GPU staging
    ComPtr<ID3D12Resource> m_nsConstantBuffer;  // b0: constant buffer

    // prediction kernel
    ComPtr<ID3D12PipelineState> m_psoPrediction;

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

    // third pass: reorder
    ComPtr<ID3D12PipelineState> m_psoReorder;
    ComPtr<ID3D12Resource> m_nsParticlesOut;

    // lambda calculation kernel
    ComPtr<ID3D12PipelineState> m_psoComputeLambda;

    // delta kernel
    ComPtr<ID3D12PipelineState> m_psoComputeDelta;

    // solve collision constraints kernel 
    ComPtr<ID3D12PipelineState> m_psoCollisionConstraints;

    // apply velocity
    ComPtr<ID3D12PipelineState> m_psoUpdateVelocity;

    // xsph kernel
    ComPtr<ID3D12PipelineState> m_psoComputeXSPH;

    // kernel that performs all the final operations
    // TODO: make this work lol
    ComPtr<ID3D12PipelineState> m_psoComputeFinalize;

    // readback
    ComPtr<ID3D12Resource> m_nsReadbackParticlesIn;

    // testing
    ComPtr<ID3D12Resource> m_computeReadbackBuffer;
    
};
