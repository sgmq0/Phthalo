#include "stdafx.h"
#include "ParticleSystem.h"

ParticleSystem::ParticleSystem() :
    m_numParticles(100)
{
}

ParticleSystem::ParticleSystem(UINT numParticles) :
    m_numParticles(numParticles)
{  
}

void ParticleSystem::LoadParticles(std::vector<InstanceData>& instances)
{
	// initializes all the particles
	float spacing = 0.3f;
    int perAxis = (int)cbrt(NUM_PARTICLES);
    int i = 0;
    for (int x = 0; x < perAxis; x++)
    for (int y = 0; y < perAxis; y++)
    for (int z = 0; z < perAxis; z++) {
        if (i >= NUM_PARTICLES) break;
        m_particles[i].position = {
            (x - perAxis/2.0f) * spacing,
            y * spacing + 1.0f,   // start above floor
            (z - perAxis/2.0f) * spacing
        };
        m_particles[i].velocity = {0, 0, 0};
        i++;
    }
    
    instances.resize(NUM_PARTICLES);
}