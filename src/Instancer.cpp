#include "stdafx.h"
#include "Instancer.h"

Instancer::Instancer() :
    m_sphere(SphereMesh(0.1f))
{
    m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();
}
