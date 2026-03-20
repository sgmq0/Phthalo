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

    static const UINT NUM_PARTICLES = 1000;
    Particle m_particles[NUM_PARTICLES];

    Instancer m_instancer;

    // test pipeline stuff
    void ParticleSystem::CreateComputePipeline(
        ID3D12Device* device,
        std::wstring shaderPath,
        ComPtr<ID3D12CommandAllocator>& commandAllocator,
        ComPtr<ID3D12GraphicsCommandList>& commandList);
    
    ComPtr<ID3D12Resource> m_computeTestBuffer; 
    ComPtr<ID3D12RootSignature> m_computeTestRootSignature;
    ComPtr<ID3D12PipelineState> m_computeTestPipeline;
	ComPtr<ID3DBlob> m_computeTestShaderBlob;

private:
    
};
