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
    void Update(float dt);
    void UpdateInstances();

    float ComputeDensityConstraint(int i, float radius);
    XMFLOAT3 ComputeGradiantConstraint(int i, int j, float density, float radius);

    UINT m_numParticles = 100;
    static const UINT NUM_PARTICLES = 1000;
    Particle m_particles[NUM_PARTICLES];

    ComPtr<ID3D12Resource> m_particleBuffer;    // STEP 1
    ComPtr<ID3D12Resource> m_lambdaBuffer;      // STEP 3-4
    ComPtr<ID3D12Resource> m_deltaBuffer;       // STEP 5
    ComPtr<ID3D12Resource> m_densityBuffer;  
    ComPtr<ID3D12Resource> m_xsphBuffer; 
    ComPtr<ID3D12Resource> m_simConstantBuffer; // buffer for all the constants

    Instancer m_instancer;

    // compute pipeline things
    void ParticleSystem::CreateComputePipeline(
        ID3D12Device* device,
        std::wstring shaderPath,
        ComPtr<ID3D12CommandAllocator>& commandAllocator,
        ComPtr<ID3D12GraphicsCommandList>& commandList);
    
    ComPtr<ID3D12Resource> m_computeTestBuffer; 
    ComPtr<ID3D12RootSignature> m_computeTestRootSignature;
    ComPtr<ID3D12PipelineState> m_computeTestPipeline;
	ComPtr<ID3DBlob> m_computeTestShaderBlob;

    // testing
    ComPtr<ID3D12Resource> m_computeReadbackBuffer;

    // compute for gpu grid search
    static const int   GRID_DIM        = 7;     // ceil(boxSize*2 / cellSize) + 1
    static const int   GRID_TOTAL_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;  // 343
    static const float GRID_CELL_SIZE;   // defined in .cpp, equals smoothing radius
    static const float GRID_ORIGIN;      // defined in .cpp, e.g. -3.5f

    struct GridEntry { UINT cellIndex; UINT particleIndex; };

    // GPU resources
    ComPtr<ID3D12Resource> m_positionBuffer;       // float4[NUM_PARTICLES] — input
    ComPtr<ID3D12Resource> m_gridEntryBuffer;      // GridEntry[NUM_PARTICLES] — output

    // pipeline objects  
    ComPtr<ID3D12RootSignature> m_assignRootSignature;
    ComPtr<ID3D12PipelineState> m_assignPipeline;
    ComPtr<ID3D12Resource>      m_assignConstantBuffer;

    void CreateAssignPipeline(ID3D12Device* device, std::wstring shaderPath);
    void UploadPositions(ID3D12GraphicsCommandList* cmdList);
    void DispatchAssign(ID3D12GraphicsCommandList* cmdList);

private:
    
};
