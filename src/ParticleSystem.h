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
        std::wstring mcShaderPath,
        ComPtr<ID3D12CommandAllocator>& commandAllocator,
        ComPtr<ID3D12GraphicsCommandList>& commandList
    );

    // update calls
    void ReadbackParticleData(ID3D12GraphicsCommandList* cmdList);  // load back the particles to CPU
    void ReadbackVertexData(ID3D12GraphicsCommandList* cmdList);

    void CopyBackResources(ID3D12GraphicsCommandList* cmdList);
    void UpdatePBD(float dt, ID3D12GraphicsCommandList* cmdList);
    void UpdateInstances();

    // compute dispatches
    void DispatchGPUCommands(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchInit(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchPrediction(ID3D12GraphicsCommandList* cmdList, float dt);
    void DispatchNeighborSearch(ID3D12GraphicsCommandList* cmdList);

    void DispatchMarchingCubes(ID3D12GraphicsCommandList* cmdList);
    void DispatchMCInit(ID3D12GraphicsCommandList* cmdList);

    // various getters for private variables
    ComPtr<ID3D12PipelineState> GetPsoClear();
    ID3D12Resource* GetMCVertexBuffer() const { return m_mcVertexBuffer.Get(); }
    ID3D12Resource* GetMCArgBuffer() const { return m_mcIndirectArgs.Get(); }

    // instancing member variables
    std::vector<Particle> m_particles;
    Instancer m_instancer;

    std::vector<Vertex> m_vertices;

    // --------- CONSTANTS --------
    static const UINT NUM_X = 60;
    static const UINT NUM_Y = 60;
    static const UINT NUM_Z = 15;
    static const UINT NUM_PARTICLES = NUM_X * NUM_Y * NUM_Z;
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
    const int ITERATIONS = 3;

    // uniform grid search consts
    const UINT NS_DIM_X = (UINT)ceil((BBOX_SIZE_XZ * 2) / CELL_SIZE) + 2;
    const UINT NS_DIM_Y = (UINT)ceil(BBOX_SIZE_Y / CELL_SIZE) + 2;
    const UINT NS_DIM_Z = (UINT)ceil((BBOX_SIZE_XZ * 2) / CELL_SIZE) + 2;
    const UINT NS_NUM_CELLS = NS_DIM_X * NS_DIM_Y * NS_DIM_Z;

    // marching cube grid consts
    const UINT MC_DIM_X = 64;
    const UINT MC_DIM_Y = 128;
    const UINT MC_DIM_Z = 64;
    const UINT MC_NUM_CELLS = MC_DIM_X * MC_DIM_Y * MC_DIM_Z;
    const float MC_CELL_SIZE = (BBOX_SIZE_XZ * 2.0f) / MC_DIM_X;
    const float MC_ISO = 5.0f; //isosurface threshold
    const UINT MC_MAX_TRIS = MC_NUM_CELLS * 5;
    
    bool m_nsFirstFrame = true;

private:

    // ----- upload, readback, etc.
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12Resource> m_nsUploadBuffer;    // CPU -> GPU staging
    ComPtr<ID3D12Resource> m_nsReadbackParticlesIn;
    ComPtr<ID3D12Resource> m_mcReadbackVertexBuffer;
    ComPtr<ID3D12Resource> m_mcReadbackArgs;
    ComPtr<ID3D12Resource> m_nsConstantBuffer;  // b0: constant buffer
    ComPtr<ID3D12Resource> m_mcConstantBuffer;  // b1: constant buffer for marching cubes

    // ----- resources for uniform grid search -----
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

    // ----- resources for pbf -----
    ComPtr<ID3D12PipelineState> m_psoPrediction;
    ComPtr<ID3D12PipelineState> m_psoComputeLambda;     
    ComPtr<ID3D12PipelineState> m_psoComputeDelta;
    ComPtr<ID3D12PipelineState> m_psoCollisionConstraints;
    ComPtr<ID3D12PipelineState> m_psoUpdateVelocity;    // TODO: make this work lol
    ComPtr<ID3D12PipelineState> m_psoComputeXSPH;

    // all of the resources for marching cubes
    ComPtr<ID3D12Resource> m_mcScalarField;       // float per grid vertex
    ComPtr<ID3D12Resource> m_mcVertexBuffer;      // output triangles
    ComPtr<ID3D12Resource> m_mcIndirectArgs;      // DrawInstanced args
    ComPtr<ID3D12Resource> m_mcVertexCounter;     // atomic counter
    ComPtr<ID3D12PipelineState> m_psoClearArgs;
    ComPtr<ID3D12PipelineState> m_psoClearField;
    ComPtr<ID3D12PipelineState> m_psoBuildField;
    ComPtr<ID3D12PipelineState> m_psoMarchingCubes;

    // resources for sdf generation
    ComPtr<ID3D12Resource> m_nsSDFVolume;
};
