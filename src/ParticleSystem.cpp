#include "stdafx.h"
#include "ParticleSystem.h"

ParticleSystem::ParticleSystem() :
    m_numParticles(100),
    m_instancer(Instancer())
{
}

ParticleSystem::ParticleSystem(UINT numParticles) :
    m_numParticles(numParticles),
    m_instancer(Instancer())
{  
}

void ParticleSystem::LoadParticles()
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
    
    m_instancer.m_instances.resize(NUM_PARTICLES);
}

float Poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 >= h2) return 0.0f;
    float x = h2 - r2;
    return (315.0f / (64.0f * XM_PI * pow(h, 9))) * x * x * x;
}

XMFLOAT3 SpikyGradient(XMFLOAT3 ri, XMFLOAT3 rj, float h) {
    float dx = ri.x - rj.x;
    float dy = ri.y - rj.y;
    float dz = ri.z - rj.z;
    float r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-6f || r >= h) return {0, 0, 0};
    float coeff = -45.0f / (XM_PI * pow(h, 6)) * (h - r) * (h - r) / r;
    return { coeff * dx, coeff * dy, coeff * dz };
}

void ParticleSystem::SolveConstraints(float dist, float distSquared) {
    const float rho0    = 10.0f;
    const float epsilon = 1000.0f;
    const int   iters   = 3;

    for (int iter = 0; iter < iters; iter++) {

        // compute lambda for each particle
        for (int i = 0; i < NUM_PARTICLES; i++) {
            // estimate density via Poly6
            //float rho = Poly6(0.0f, dist);  // self contribution
			float rho = rho0;

            for (int j : m_particles[i].neighbors) {
                float dx = m_particles[i].predictedPosition.x - m_particles[j].predictedPosition.x;
                float dy = m_particles[i].predictedPosition.y - m_particles[j].predictedPosition.y;
                float dz = m_particles[i].predictedPosition.z - m_particles[j].predictedPosition.z;
                rho += Poly6(dx*dx + dy*dy + dz*dz, dist);
            }

            // density constraint C_i = rho/rho0 - 1
            float C = rho / rho0 - 1.0f;

            // sum of squared gradients for denominator
            float sumGrad2 = 0.0f;
            for (int n = 0; n < m_particles[i].neighbors.size(); n++) {
                int j = m_particles[i].neighbors[n];
                XMFLOAT3 grad = SpikyGradient(m_particles[i].predictedPosition, m_particles[j].predictedPosition, dist);
                float g = (grad.x*grad.x + grad.y*grad.y + grad.z*grad.z) / rho0;
                sumGrad2 += g;
            }

            m_particles[i].lambda = -C / (sumGrad2 + epsilon);
        }

        // compute and apply delta p
        std::vector<XMFLOAT3> deltas(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});

		for (int i = 0; i < NUM_PARTICLES; i++) {
			for (int j : m_particles[i].neighbors) {
				float w = m_particles[i].lambda + m_particles[j].lambda;
				XMFLOAT3 grad = SpikyGradient(m_particles[i].predictedPosition, m_particles[j].predictedPosition, dist);
				deltas[i].x += w * grad.x / rho0;
				deltas[i].y += w * grad.y / rho0;
				deltas[i].z += w * grad.z / rho0;
			}
		}

		for (int i = 0; i < NUM_PARTICLES; i++) {
			m_particles[i].predictedPosition.x += deltas[i].x;
			m_particles[i].predictedPosition.y += deltas[i].y;
			m_particles[i].predictedPosition.z += deltas[i].z;
		}
    }
}

void ParticleSystem::Update(float dt) {
    if (dt <= 0.0f || dt > 0.1f) return;

	const float dist = 1.0f;
	const float distSquared = dist * dist;

	for (int i = 0; i < NUM_PARTICLES; i++) {

		// apply forces
		m_particles[i].velocity.y += -9.8f * dt;

		// predict position
		m_particles[i].predictedPosition.x = m_particles[i].position.x + dt * m_particles[i].velocity.x;
		m_particles[i].predictedPosition.y = m_particles[i].position.y + dt * m_particles[i].velocity.y;
		m_particles[i].predictedPosition.z = m_particles[i].position.z + dt * m_particles[i].velocity.z;
    }

	// find neighbors
	for (int i = 0; i < NUM_PARTICLES; i++) {
		m_particles[i].neighbors.clear();
	}

	for (int i = 0; i < NUM_PARTICLES; i++) {
		for (int j = i + 1; j < NUM_PARTICLES; j++) {
			float dx = m_particles[i].predictedPosition.x - m_particles[j].predictedPosition.x;
			float dy = m_particles[i].predictedPosition.y - m_particles[j].predictedPosition.y;
			float dz = m_particles[i].predictedPosition.z - m_particles[j].predictedPosition.z;
			float dist2 = dx*dx + dy*dy + dz*dz;

			if (dist2 < distSquared) {
				m_particles[i].neighbors.push_back(j);
				m_particles[j].neighbors.push_back(i);
			}
		}
	}

	// calculations for all particles
	SolveConstraints(dist, distSquared);

	// update positions and velocity
	for (int i = 0; i < NUM_PARTICLES; i++) {
		if (m_particles[i].predictedPosition.y < 0.0f) {
			m_particles[i].predictedPosition.y = 0.0f;
		}

		const float boxMin = -2.0f;
		const float boxMax =  2.0f;

		if (m_particles[i].predictedPosition.x < boxMin) { m_particles[i].predictedPosition.x = boxMin; }
		if (m_particles[i].predictedPosition.x > boxMax) { m_particles[i].predictedPosition.x = boxMax; }
		if (m_particles[i].predictedPosition.y < boxMin) { m_particles[i].predictedPosition.y = boxMin; }
		if (m_particles[i].predictedPosition.z < boxMin) { m_particles[i].predictedPosition.z = boxMin; }
		if (m_particles[i].predictedPosition.z > boxMax) { m_particles[i].predictedPosition.z = boxMax; }

		m_particles[i].velocity.x = (m_particles[i].predictedPosition.x - m_particles[i].position.x) / dt;
		m_particles[i].velocity.y = (m_particles[i].predictedPosition.y - m_particles[i].position.y) / dt;
		m_particles[i].velocity.z = (m_particles[i].predictedPosition.z - m_particles[i].position.z) / dt;
		m_particles[i].position = m_particles[i].predictedPosition;

		if (m_particles[i].position.y < 0.0f) {
			m_particles[i].velocity.y *= -0.3f; // dampen bounce
		}
	}

	UpdateInstances();
}

void ParticleSystem::UpdateInstances() {
    // push to instances vector
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMMATRIX mat = XMMatrixTranslation(
            m_particles[i].position.x,
            m_particles[i].position.y,
            m_particles[i].position.z
        );

        XMStoreFloat4x4(&m_instancer.m_instances[i].worldMatrix, mat);
    }

    memcpy(m_instancer.m_pInstanceDataBegin, m_instancer.m_instances.data(), sizeof(InstanceData) * NUM_PARTICLES);
}
