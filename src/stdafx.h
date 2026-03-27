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

struct Particle {
    XMFLOAT3 position;
    XMFLOAT3 predictedPosition;
    XMFLOAT3 velocity;
    float density;
    float lambda;
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
};

struct NSConstants {
    XMFLOAT3 gridOrigin;
    float cellSize;
    int gridDimX;
    int gridDimY;
    int gridDimZ;
    int numParticles;
    int numCells;
    int _pad[3];
};

