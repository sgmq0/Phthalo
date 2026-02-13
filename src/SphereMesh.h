#pragma once
#include "stdafx.h"

using namespace DirectX;

class SphereMesh {
public:
    SphereMesh(float rad) { m_radius = rad; }

    // vertex struct, self-explanatory.
    struct Vertex
	{
		XMFLOAT3 position;
		XMFLOAT4 color;
	};

    float m_radius = 1.0f;
    int m_stacks = 16;
    int m_slices = 16;

    // populates the vertex and index arrays
    void LoadMesh() {
        for (int i = 0; i <= m_stacks; i++) {
            float phi = XM_PI * ((float)i / m_stacks);        // 0 to PI (top to bottom)

            for (int j = 0; j <= m_slices; j++) {
                float theta = XM_2PI * ((float)j / m_slices); // 0 to 2PI (around)

                Vertex v;

                v.position = {
                    m_radius * sinf(phi) * cosf(theta),
                    m_radius * cosf(phi),
                    m_radius * sinf(phi) * sinf(theta)
                };

                // normal is just the normalized position for a unit sphere
                v.color = {1.0f, 0.0f, 0.0f, 1.0f};

                sphereVertices.push_back(v);
            }
        }

        // generate indices
        for (int i = 0; i < m_stacks; i++) {
            for (int j = 0; j < m_slices; j++) {
                int row1 = i * (m_slices + 1);
                int row2 = (i + 1) * (m_slices + 1);

                // two triangles per quad
                sphereIndices.push_back(row1 + j);
                sphereIndices.push_back(row1 + j + 1);
                sphereIndices.push_back(row2 + j + 1);

                sphereIndices.push_back(row1 + j);
                sphereIndices.push_back(row2 + j + 1);
                sphereIndices.push_back(row2 + j);
            }
        }
    };

    std::vector<Vertex> sphereVertices;
    std::vector<UINT32> sphereIndices;

};
