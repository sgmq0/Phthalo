#pragma once
#include "stdafx.h"

using namespace DirectX;

class ParticleSystem {
public:
    ParticleSystem();
    ParticleSystem(UINT numParticles);

    void LoadParticles(std::vector<InstanceData>& instances);
    void Update(float dt, std::vector<InstanceData>& instances);
    void SolveConstraints(float dist, float distSquared);

    UINT m_numParticles = 100;
    static const UINT NUM_PARTICLES = 1000;
    Particle m_particles[NUM_PARTICLES];

private:
    
};
