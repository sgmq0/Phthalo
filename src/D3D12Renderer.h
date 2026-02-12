#pragma once

#include "DXApplication.h"
#include "Camera.h"

using namespace DirectX;

using Microsoft::WRL::ComPtr;

class D3D12Renderer : public DXApplication {
public:
	D3D12Renderer(UINT width, UINT height);

    // inherits from DXSample, this forms the standard render loop.
	virtual void OnInit();
	virtual void OnUpdate();
	virtual void OnRender();
	virtual void OnDestroy();

private:
    // use double buffering - two render targets.
	static const UINT FrameCount = 2;

    // vertex struct, self-explanatory.
	struct Vertex
	{
		XMFLOAT3 position;
		XMFLOAT4 color;
	};

	// ----- pipeline objects created in LoadPipeline() -----

    // these define the renderable area of the window.
    // they're both set in the constructor.
    CD3DX12_VIEWPORT m_viewport;    // maps 3D coordinates to screen space
    CD3DX12_RECT m_scissorRect;     // clips rendering to a specific rectangle.

    // the swapchain manages the front and back buffers. 
    ComPtr<IDXGISwapChain3> m_swapChain;

    // the representation of the GPU.
    ComPtr<ID3D12Device> m_device;

    // these are the two buffers used in our swapchain.
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];

    // command list things.
    ComPtr<ID3D12GraphicsCommandList> m_commandList;    // records gpu commands -> submitted to command queue.
    ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];  // manages memory for gpu commands.
    ComPtr<ID3D12CommandQueue> m_commandQueue;          // submits commands recorded by m_commandList.

    // descriptors for the render targets. 
    // we need ways to describe the render targets to the GPU.
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    UINT m_rtvDescriptorSize;   // just the size in bytes of a descriptor. used for offsetting

    // ----- asset objects created in LoadAssets(). -----

    // a root signature defines what resources your shaders can access.
    // information about shader parameters, essentially.
    ComPtr<ID3D12RootSignature> m_rootSignature;

    // contains the entire pipeline configuration.
    ComPtr<ID3D12PipelineState> m_pipelineState;

    // ----- App resources. ----
    ComPtr<ID3D12Resource> m_vertexBuffer;          // buffer holding vertex data
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;    // describes the vertex buffer layout for the gpu.

    ComPtr<ID3D12Resource> m_indexBuffer;           // ditto, for index buffer
    D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

    // ----- Synchronization objects. -----
    UINT m_frameIndex;              // tracks the currently used frame buffer (0 or 1)
                                    // and uses it to index into m_renderTargets.

    // fences! yay!
    ComPtr<ID3D12Fence> m_fence;    // used to synchronize the gpu and cpu.
                                    // ensures that the gpu has finished rendering a frame before
                                    // its modified by the cpu.
    HANDLE m_fenceEvent;            // event that waits for gpu work to complete.
                                    // apparently this isnt best practice, look into optimizations?
    UINT64 m_fenceValues[FrameCount];            // counter (for now)

    void LoadPipeline();            // loads and initializes the core pipeline objects. 
    void LoadAssets();              // loads shaders and creates the pipeline state as well as vertex data.
    void PopulateCommandList();     // records commands into the command list.

    // stuff to replace WaitForPreviousFrame()
    void MoveToNextFrame();
    void WaitForGPU();

    // ----- Camera stuff -----
    ComPtr<ID3D12Resource> m_constantBuffer;
    UINT8* m_pCbvDataBegin = nullptr;  // persistent mapped pointer

    struct MVPConstantBuffer {
        XMFLOAT4X4 mvp;
    };
    MVPConstantBuffer m_cbData;

    Camera m_camera;
    UINT64 m_lastFrameTime = 0;

    void D3D12Renderer::OnKeyDown(UINT8 key) { m_camera.OnKeyDown(key); }
    void D3D12Renderer::OnKeyUp  (UINT8 key) { m_camera.OnKeyUp(key);   }
    void D3D12Renderer::OnMouseMove(int dx, int dy) { m_camera.OnMouseMove(dx, dy); }
};