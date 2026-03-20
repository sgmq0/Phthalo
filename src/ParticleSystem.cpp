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


void ParticleSystem::Update(float dt) {
    if (dt <= 0.0f || dt > 0.1f) return;

	const float radius      = 1.0f;
    const float radius2     = radius * radius;
    const float rho_0       = 10.0f;       // rest density
    const float epsilon     = 1000.0f;
    const float damping     = 0.999f;
    const float boxSize     = 3.0f;
    const float restitution = 0.3f;
    const float viscosity   = 0.05f;
    const int iterations    = 3;

    // ------------ STEP 1: PREDICT POSITIONS --------------
	for (int i = 0; i < NUM_PARTICLES; i++) {

		// apply forces
		m_particles[i].velocity.y += -9.8f * dt;

		// predict position
		m_particles[i].predictedPosition.x = m_particles[i].position.x + dt * m_particles[i].velocity.x;
		m_particles[i].predictedPosition.y = m_particles[i].position.y + dt * m_particles[i].velocity.y;
		m_particles[i].predictedPosition.z = m_particles[i].position.z + dt * m_particles[i].velocity.z;
    }

	// ------------ STEP 2: FIND NEIGHBORS --------------
	for (int i = 0; i < NUM_PARTICLES; i++) {
		m_particles[i].neighbors.clear();
	}
    
	for (int i = 0; i < NUM_PARTICLES; i++) {
		for (int j = i + 1; j < NUM_PARTICLES; j++) {
			float dx = m_particles[i].predictedPosition.x - m_particles[j].predictedPosition.x;
			float dy = m_particles[i].predictedPosition.y - m_particles[j].predictedPosition.y;
			float dz = m_particles[i].predictedPosition.z - m_particles[j].predictedPosition.z;
			float dist2 = dx*dx + dy*dy + dz*dz;

			if (dist2 < radius2) {
				m_particles[i].neighbors.push_back(j);
				m_particles[j].neighbors.push_back(i);
			}
		}
	}

	// calculations for all particles
    for (int iter = 0; iter < iterations; iter++) {

        // for all particles, calculate lambda_i
        for (int i = 0; i < NUM_PARTICLES; i++) {
            
            // ------------ STEP 3 : COMPUTE DENSITIES rho_i --------------
            float rho_i = ComputeDensityConstraint(i, radius);

            // ------------ STEP 4 : COMPUTE LAMBDA --------------
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

        // ------------ STEP 5 : COMPUTE DENSITIES DELTA RHO --------------
        std::vector<XMFLOAT3> deltas(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});

        for (int i = 0; i < NUM_PARTICLES; i++) {
            Particle p = m_particles[i];

            // artifical tensile pressure correction constants
            const float corr_n = 4.0;
            const float corr_h = 0.30;
            const float corr_k = 1e-04;
            const float corr_w = Poly6(XMFLOAT3(corr_h * radius, 0.0, 0.0), radius);

            float sum_x = 0.0;
            float sum_y = 0.0;
            float sum_z = 0.0;

            for (int j : p.neighbors) {
                Particle p_neighbor = m_particles[j];

                // artificial tensile pressure correction 
                const float poly = Poly6(SubtractFloat3(p.position, p_neighbor.position), radius);
                const float ratio = poly / corr_w;
                const float corr_coeff = -corr_k * pow(ratio, corr_n);
                const float coeff = p.lambda + p_neighbor.lambda + corr_coeff;

                XMFLOAT3 sum = SpikyGradient(SubtractFloat3(p.position, p_neighbor.position), radius);
                sum_x += sum.x * coeff;
                sum_y += sum.y * coeff;
                sum_z += sum.z * coeff;
            }

            float rho_coeff = 1.0 / rho_0;
            XMFLOAT3 sum_vec = XMFLOAT3(rho_coeff * sum_x, rho_coeff * sum_y, rho_coeff * sum_z);

            // solve for delta rho
            deltas[i].x += sum_vec.x;
            deltas[i].y += sum_vec.y;
            deltas[i].z += sum_vec.z;
        }

        // ------------ STEP 6 : APPLY DELTA RHO --------------
		for (int i = 0; i < NUM_PARTICLES; i++) {
			m_particles[i].predictedPosition.x += deltas[i].x;
			m_particles[i].predictedPosition.y += deltas[i].y;
			m_particles[i].predictedPosition.z += deltas[i].z;
		}
    }

    std::vector<float> densities(NUM_PARTICLES, 0.0f);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        densities[i] = ComputeDensityConstraint(i, radius);
    }

    // XSPH viscosity computation
    std::vector<XMFLOAT3> xsph(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j : m_particles[i].neighbors) {
            float val = Poly6(SubtractFloat3(m_particles[i].position, m_particles[j].position), radius);
            XMFLOAT3 velocity = SubtractFloat3(m_particles[j].velocity, m_particles[i].velocity);

            float coeff = 1.0f / densities[i];
            xsph[i].x += val * velocity.x;
            xsph[i].y += val * velocity.y;
            xsph[i].z += val * velocity.z;
        }
    }

	// update positions and velocity
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMFLOAT3& pred = m_particles[i].predictedPosition;

        // constraints
        pred.x = max(pred.x, -boxSize); 
        pred.x = min(pred.x, boxSize);
        pred.y = max(pred.y, 0.0f); 
        pred.y = min(pred.y, 8.0f);
        pred.z = max(pred.z, -boxSize); 
        pred.z = min(pred.z, boxSize);

        // update velocity
        m_particles[i].velocity.x = damping * (pred.x - m_particles[i].position.x) / dt;
		m_particles[i].velocity.y = damping * (pred.y - m_particles[i].position.y) / dt;
		m_particles[i].velocity.z = damping * (pred.z - m_particles[i].position.z) / dt;

        // apply XSPH velocity
        m_particles[i].velocity.x += xsph[i].x * viscosity;
        m_particles[i].velocity.y += xsph[i].y * viscosity;
        m_particles[i].velocity.z += xsph[i].z * viscosity;

        // TODO: vorticity confinement

        // update position 
		m_particles[i].position = pred;
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
