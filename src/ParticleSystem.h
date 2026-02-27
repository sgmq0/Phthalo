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
    void SolveConstraints(float dist, float distSquared);

    UINT m_numParticles = 100;
    static const UINT NUM_PARTICLES = 1000;
    Particle m_particles[NUM_PARTICLES];

    Instancer m_instancer;

private:
    
};
