#include <windows.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>

using Microsoft::WRL::ComPtr;

int main()
{
    ComPtr<ID3D12Device> device;

    HRESULT hr = D3D12CreateDevice(
        nullptr,
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&device)
    );

    if (FAILED(hr))
    {
        std::cerr << "Failed to create D3D12 device\n";
        return -1;
    }

    std::cout << "DirectX 12 device created successfully!\n";
    return 0;
}
