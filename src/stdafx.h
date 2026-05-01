#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include "d3dx12.h"

#include <string>
#include <wrl.h>
#include <shellapi.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;
using Microsoft::WRL::ComPtr;

struct InstanceData {
    XMFLOAT4X4 worldMatrix;
};

struct Vertex
{
    float x, y, z, w;
    XMFLOAT4 normal;
};

struct Particle {
    XMFLOAT3 position;
    XMFLOAT3 predictedPosition;
    XMFLOAT3 velocity;
    float density;
    float lambda;
    XMFLOAT3 xsph;
    XMFLOAT3 delta;
    std::vector<int> neighbors;
};

struct GPUParticle {
    XMFLOAT3 position;
    float _pad0;
    XMFLOAT3 predictedPosition;
    float _pad1;
    XMFLOAT3 velocity;
    float density;
    float lambda;
    int originalIndex;
    float _pad2[2];
    XMFLOAT3 xsph;
    float _pad3;
    XMFLOAT3 delta;
    float _pad4;
};

struct NSConstants {
    XMFLOAT3 gridOrigin;
    float cellSize;
    int gridDimX;
    int gridDimY;
    int gridDimZ;
    int numParticles;
    int numCells;
    float dt;
    int _pad[2];
};

struct MCConstants {
    XMFLOAT3 mcOrigin; 
    float mcCellSize;
    XMINT3 mcDims;
    float mcIso;
    int mcMaxTris;
    float _pad[3];
};

struct VPConstantBuffer {
    XMFLOAT4X4 vp;
    // XMFLOAT3 camPos;
    // float _pad;
};

