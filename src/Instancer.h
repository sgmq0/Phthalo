#pragma once
#include "stdafx.h"

#include "SphereMesh.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

class Instancer {
public:
    Instancer();

    SphereMesh m_sphere;
    size_t m_sphereIndexCount;

    ComPtr<ID3D12Resource> m_instanceBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_instanceBufferView;
    UINT8* m_pInstanceDataBegin = nullptr;

    std::vector<InstanceData> m_instances;

private:

};
