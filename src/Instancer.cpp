#include "stdafx.h"
#include "Instancer.h"

Instancer::Instancer() :
    m_sphere(SphereMesh(0.1f, 10))
{
    m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();
}

Instancer::Instancer(float radius) :
    m_sphere(SphereMesh(radius, 10))
{
    m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();
}

