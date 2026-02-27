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

XMFLOAT3 SubtractFloat3(XMFLOAT3 v1, XMFLOAT3 v2) {
    return XMFLOAT3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

float SquaredNorm(XMFLOAT3 vec) {
    float squared = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    return squared;
}

float Poly6(XMFLOAT3 r, float h) {
    const float coeff = 315.0 / (64.0 * XM_PI);

    float h2 = h * h;
    float r2 = SquaredNorm(r);

    if (r2 > h2) {
        return 0.0;
    }

    float h9 = pow(h, 9);
    float diff = h2 - r2;
    float diff3 = diff * diff * diff;

    return (coeff / h9) * diff3;
}


XMFLOAT3 SpikyGradient(XMFLOAT3 r, float h) {
    float coeff = 45.0f / XM_PI;

    float norm = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);

    if (norm > h) {
        return XMFLOAT3(0.0, 0.0, 0.0);
    }

    float h6 = pow(h, 6);
    float diff = h - norm;
    float diff2 = diff * diff;

    float times = (coeff / (h6 * max(norm, 1e-24f))) * diff2;
    XMFLOAT3 result = XMFLOAT3(-r.x * times, -r.y * times, -r.z * times);
    return result;
}

float ParticleSystem::ComputeDensityConstraint(int i, float radius) {
    Particle p_target = m_particles[i];

    float density = 0.0;
    for (int j : p_target.neighbors) {
        Particle p = m_particles[j];

        XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
        density += Poly6(pos, radius);
    }

    return density;
}

XMFLOAT3 ParticleSystem::ComputeGradiantConstraint(int i, int j, float density, float radius) {
    Particle p_target = m_particles[i];

    if (i == j) {
        XMFLOAT3 sum = XMFLOAT3(0.0, 0.0, 0.0);
        for (int neighbor : m_particles[i].neighbors) {
            Particle p = m_particles[neighbor];

            XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
            XMFLOAT3 grad = SpikyGradient(pos, radius);
            sum.x += grad.x;
            sum.y += grad.y;
            sum.z += grad.z;
        }

        XMFLOAT3 result = XMFLOAT3(sum.x / density, sum.y / density, sum.z / density);
        return result;
    } else {
        Particle p = m_particles[j];

        XMFLOAT3 pos = SubtractFloat3(p_target.predictedPosition, p.predictedPosition);
        XMFLOAT3 grad = SpikyGradient(pos, radius);
        return XMFLOAT3(-grad.x, -grad.y, -grad.z);
    }
}

void ParticleSystem::SolveConstraints(float dist, float distSquared) {
    const float rho_0    = 10.0f;       // rest density
    const float radius   = dist;
    const float epsilon  = 1000.0f;
    const int iterations = 3;

    for (int iter = 0; iter < iterations; iter++) {

        // for all particles, calculate lambda_i
        for (int i = 0; i < NUM_PARTICLES; i++) {
            
            // estimate density with Poly6
            float rho_i = ComputeDensityConstraint(i, radius);

            float constraint = (rho_i / rho_0) - 1.0; // numerator

            float denominator = 0.0;
            for (int j : m_particles[i].neighbors) {
                XMFLOAT3 gradConstraint = ComputeGradiantConstraint(i, j, rho_0, radius);
                denominator += SquaredNorm(gradConstraint);
            }

            // add an epislon value for relaxation
            denominator += epsilon;
            m_particles[i].lambda = -constraint / denominator;
        }

        // TODO: for all particles, calculate delta p

        // compute and apply delta p
        std::vector<XMFLOAT3> deltas(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});

		for (int i = 0; i < NUM_PARTICLES; i++) {
			for (int j : m_particles[i].neighbors) {
				float w = m_particles[i].lambda + m_particles[j].lambda;
                XMFLOAT3 pos = SubtractFloat3(m_particles[i].predictedPosition, m_particles[j].predictedPosition);
				XMFLOAT3 grad = SpikyGradient(pos, radius);
				deltas[i].x += w * grad.x / rho_0;
				deltas[i].y += w * grad.y / rho_0;
				deltas[i].z += w * grad.z / rho_0;
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
