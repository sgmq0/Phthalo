#pragma once
#include "stdafx.h"

#include "Instancer.h"

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

private:
    
};
