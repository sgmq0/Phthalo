
#include "stdafx.h"
#include "D3D12Renderer.h"
#include <random>

std::mt19937 rng(std::random_device{}());

D3D12Renderer::D3D12Renderer(UINT width, UINT height) :
	DXApplication(width, height, L"Renderer"),
	m_frameIndex(0),
	m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)),
	m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
	m_fenceValues{},
	m_rtvDescriptorSize(0),
	m_camera(XMFLOAT3(-10.0f, 2.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f),
	m_particleSystem(ParticleSystem(500))
{
}

void D3D12Renderer::OnInit()
{
	LoadPipeline();
	m_particleSystem.LoadParticles();
	LoadAssets();
}

void D3D12Renderer::OnUpdate()
{
	// update camera and stuff here
	UINT64 now = GetTickCount64();
    float dt = (m_lastFrameTime == 0) ? 0.0f : (now - m_lastFrameTime) / 1000.0f;
    m_lastFrameTime = now;

    m_camera.Update(dt);

    // build mvp
    XMMATRIX model = XMMatrixIdentity();
    XMMATRIX view  = m_camera.GetViewMatrix();
    XMMATRIX proj  = m_camera.GetProjectionMatrix(static_cast<float>(m_width) / m_height);
	
    XMMATRIX vp   = XMMatrixTranspose(view * proj);
    XMStoreFloat4x4(&m_cbData.vp, vp);
    memcpy(m_pCbvDataBegin, &m_cbData, sizeof(m_cbData));

	// write per instance data
	// change this into particles...

	m_particleSystem.Update(dt);

    // reset command list 
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr));

	// run the compute
	m_particleSystem.m_instancer.ComputeInstances(
		m_commandList.Get(),
		m_commandQueue.Get(),
		m_particleSystem.m_particles,
		m_particleSystem.NUM_PARTICLES);

	// execute and wait so compute is done before PopulateCommandList runs
	// this feels kind of scuffed
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* lists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, lists);
    WaitForGPU();

	static float fpsTimer = 0.0f;
	fpsTimer += dt;
	if (fpsTimer >= 0.5f)  // update twice per second so it's readable
	{
		char buf[64];
		sprintf_s(buf, "%.2f ms  |  %.0f fps", dt * 1000.0f, 1.0f / dt);
		SetCustomWindowText(std::wstring(buf, buf + strlen(buf)).c_str());
		fpsTimer = 0.0f;
	}
}

void D3D12Renderer::OnRender()
{
	DispatchCompute();

	// static bool readbackDone = false;
    // if (!readbackDone)
    // {
    //     ReadbackCompute();
    //     readbackDone = true;
    // }

	// add all the rendering commands into our command list...
	PopulateCommandList();

	// ...then execute it.
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	// present the frame
	ThrowIfFailed(m_swapChain->Present(1, 0));

	MoveToNextFrame();
}

void D3D12Renderer::OnDestroy()
{
	// make sure we wait so we don't reference destroyed resources on the GPU
	WaitForGPU();

	CloseHandle(m_fenceEvent);
}

void D3D12Renderer::LoadPipeline()
{
	UINT dxgiFactoryFlags = 0;

	// create the debug layer, only if in debug
#if defined(_DEBUG)
	ComPtr<ID3D12Debug> debugController;
	if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
		debugController->EnableDebugLayer();

		// enables debugging for the factory itself
		dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
	}
#endif

	// factory creation: This is basically the hardware interface layer.
	ComPtr<IDXGIFactory6> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	// creates device aka our gpu
	ComPtr<IDXGIAdapter1> hardwareAdapter;	// eventually change this to V4?
	GetHardwareAdapter(factory.Get(), &hardwareAdapter);

	ThrowIfFailed(D3D12CreateDevice(
		hardwareAdapter.Get(),
		D3D_FEATURE_LEVEL_11_0,
		IID_PPV_ARGS(&m_device)
	));

	// create command queue + command allocator
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

	// also create command queue for the entire compute pipeline
	D3D12_COMMAND_QUEUE_DESC computeQueueDesc = {};
	computeQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	computeQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	ThrowIfFailed(m_device->CreateCommandQueue(&computeQueueDesc, IID_PPV_ARGS(&m_computeCommandQueue)));

	// create swap chain
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = FrameCount;
	swapChainDesc.Width = m_width;
	swapChainDesc.Height = m_height;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;

	ComPtr<IDXGISwapChain1> swapChain;
	ThrowIfFailed(factory->CreateSwapChainForHwnd(
		m_commandQueue.Get(),
		Win32Application::GetHwnd(),
		&swapChainDesc,
		nullptr,
		nullptr,
		&swapChain
	));

	// upgrade to IDXGISwapChain3 lol
	ThrowIfFailed(swapChain.As(&m_swapChain));
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();	// also store frame index

	// create descriptor heap for the render target views (for swapchain)
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = FrameCount;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

	m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	// create frame resources (for swapchain)
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
	for (UINT n = 0; n < FrameCount; n++) {
		ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
		m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(1, m_rtvDescriptorSize);

		// create command allocator
		ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
	}

	// also, disable alt+enter full-screen transitions - too jank
	ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));
}


void D3D12Renderer::LoadAssets()
{
	// set up root signature, shaders, stencil buffer, pipeline objects, command list
	CreateGraphicsPipeline();

	// set up all the compute shaders for uniform grid method
	CreateComputePipeline();

	// create vertex, index, constant, & instancing buffers
	CreateBuffers();
	
	// ---------- synchronization objects, fences and such ----------
	// create fence and wait for the gpu.
	ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
	m_fenceValues[m_frameIndex]++;

	// also create compute fence (sob)
	ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence)));

	// create fence event handle for frame synchronization.
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}

	// wait for setup to complete, in this case
	WaitForGPU();
}

void D3D12Renderer::CreateGraphicsPipeline()
{
	// create one root descriptor visible to vertex shader at b0
	CD3DX12_ROOT_PARAMETER rootParam;
	rootParam.InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX);

	// create the root signature w camera
	CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
	rootSignatureDesc.Init(
		1, 
		&rootParam, 
		0, 
		nullptr, 
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
	);

	ComPtr<ID3DBlob> signature;
	ComPtr<ID3DBlob> error;
	ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
	ThrowIfFailed(m_device->CreateRootSignature(
		0, 
		signature->GetBufferPointer(), 
		signature->GetBufferSize(), 
		IID_PPV_ARGS(&m_rootSignature)
	));

	// now create the pipeline state - compile and load shaders
	ComPtr<ID3DBlob> vertexShader;
	ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
	// add compile flags if we're on debug mode for better shader debugging
	UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
	UINT compileFlags = 0;
#endif

	ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"shaders.hlsl").c_str(), nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
	ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"shaders.hlsl").c_str(), nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));

	// define vertex input layout
	// needs to be consistent with Vertex struct in SphereMesh.h
	D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
	{
		// per vertex data
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },

		// per instance data
		{ "INSTANCE_TRANSFORM", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1,  0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
		{ "INSTANCE_TRANSFORM", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 16, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
		{ "INSTANCE_TRANSFORM", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 32, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
		{ "INSTANCE_TRANSFORM", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 48, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
	};

	// depth stencil buffer
	D3D12_RESOURCE_DESC depthDesc = {};
	depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	depthDesc.Width = m_width;
	depthDesc.Height = m_height;
	depthDesc.DepthOrArraySize = 1;
	depthDesc.MipLevels = 1;
	depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthDesc.SampleDesc.Count = 1;
	depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = DXGI_FORMAT_D32_FLOAT;
	clearValue.DepthStencil.Depth = 1.0f;

	CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
	ThrowIfFailed(m_device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&depthDesc,
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&clearValue,
		IID_PPV_ARGS(&m_depthBuffer)));

	// create a descriptor heap for the DSV
	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	ThrowIfFailed(m_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));

	// create the DSV
	m_device->CreateDepthStencilView(m_depthBuffer.Get(), nullptr, 
		m_dsvHeap->GetCPUDescriptorHandleForHeapStart());

	// describe the graphics pipeline state object (PSO)
	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
	psoDesc.pRootSignature = m_rootSignature.Get();
	psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
	psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT); // no transparency by default
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psoDesc.SampleDesc.Count = 1;
	psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT); // enable depth stencil
	psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

	// create the pipeline state also
	ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));

	// create the command list
	ThrowIfFailed(m_device->CreateCommandList(0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_commandAllocators[m_frameIndex].Get(),
		m_pipelineState.Get(),
		IID_PPV_ARGS(&m_commandList)
	));

	// close it for now because main loop expects it to be
	// nothing to record yet.
	ThrowIfFailed(m_commandList->Close());
}

void D3D12Renderer::CreateComputePipeline()
{
	// 1. initialize root signature
	CD3DX12_ROOT_PARAMETER params[1];
    params[0].InitAsUnorderedAccessView(0); // u0 - one constant for now

	CD3DX12_ROOT_SIGNATURE_DESC rootDesc = {};
	rootDesc.NumParameters = 1;
	rootDesc.pParameters = params;
	rootDesc.NumStaticSamplers = 0;
	rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

	ComPtr<ID3DBlob> sig, err;
	ThrowIfFailed(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
	ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(),
		sig->GetBufferSize(), IID_PPV_ARGS(&m_computeTestRootSignature)));

	// 2. compile shaders
	ComPtr<ID3DBlob> computeShader;
	ComPtr<ID3DBlob> computeError;
	HRESULT hr = D3DCompileFromFile(GetAssetFullPath(L"gridcount.hlsl").c_str(), nullptr, nullptr, "CSGridCount", "cs_5_0", 0, 0, &computeShader, &computeError);
	if (computeError)
		OutputDebugStringA((char*)computeError->GetBufferPointer());
	ThrowIfFailed(hr);

	// 3. create pipeline state
	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = m_computeTestRootSignature.Get();
	psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
	ThrowIfFailed(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_computeTestPipeline)));

	// create compute buffer, put this into createBuffers() later
	UINT byteSize = m_particleSystem.NUM_PARTICLES * sizeof(XMFLOAT4);

	auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
		byteSize,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
	);

	CD3DX12_HEAP_PROPERTIES defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,     
        &bufferDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_computeTestBuffer)));

	// test: readback the stuff written by the compute shader
	// auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
	// CD3DX12_HEAP_PROPERTIES readbackHeap(D3D12_HEAP_TYPE_READBACK);
	// ThrowIfFailed(m_device->CreateCommittedResource(
	// 	&readbackHeap,
	// 	D3D12_HEAP_FLAG_NONE,
	// 	&readbackDesc,
	// 	D3D12_RESOURCE_STATE_COPY_DEST,
	// 	nullptr,
	// 	IID_PPV_ARGS(&m_computeReadbackBuffer)
	// ));

	// ---- compute command list ----
    ThrowIfFailed(m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_computeAllocator)));
    ThrowIfFailed(m_device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_computeAllocator.Get(),
        m_computeTestPipeline.Get(),
        IID_PPV_ARGS(&m_computeCommandList)));
    ThrowIfFailed(m_computeCommandList->Close());
}

void D3D12Renderer::CreateBuffers()
{
	// ---------- create the vertex buffer ----------
	{
		const UINT vertexBufferSize = m_particleSystem.m_instancer.m_sphere.sphereVertices.size() * sizeof(SphereMesh::Vertex);

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_vertexBuffer)));

		// copy vertex data to vertex buffer
		UINT8* pVertexDataBegin;
		CD3DX12_RANGE readRange(0, 0);
		ThrowIfFailed(m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
		memcpy(pVertexDataBegin, m_particleSystem.m_instancer.m_sphere.sphereVertices.data(), vertexBufferSize);
		m_vertexBuffer->Unmap(0, nullptr);

		// init vertex buffer view
		m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
		m_vertexBufferView.StrideInBytes = sizeof(SphereMesh::Vertex);
		m_vertexBufferView.SizeInBytes = vertexBufferSize;
	}

	// ---------- create the index buffer ----------
	{
		const UINT indexBufferSize = m_particleSystem.m_instancer.m_sphere.sphereIndices.size() * sizeof(UINT32);

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(indexBufferSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_indexBuffer)));

		// copy index data to index buffer
		UINT8* pIndexDataBegin;
		CD3DX12_RANGE readRange(0, 0);
		ThrowIfFailed(m_indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pIndexDataBegin)));
		memcpy(pIndexDataBegin, m_particleSystem.m_instancer.m_sphere.sphereIndices.data(), indexBufferSize);
		m_indexBuffer->Unmap(0, nullptr);

		// init index buffer view
		m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
		m_indexBufferView.Format = DXGI_FORMAT_R32_UINT;  // 32-bit indices, for lots of verts
		m_indexBufferView.SizeInBytes = indexBufferSize;
	}

	// ---------- constant buffer (MVP matrix) ----------
	{
		const UINT cbSize = (sizeof(VPConstantBuffer) + 255) & ~255;  // must be 256-byte aligned

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBuffer)));

		CD3DX12_RANGE readRange(0, 0);
		ThrowIfFailed(m_constantBuffer->Map(0, &readRange,
			reinterpret_cast<void**>(&m_pCbvDataBegin)));
	}

	// ---------- create the instancing buffer ----------
	// put this in instancer class
	m_particleSystem.m_instancer.Init(m_device.Get(), m_particleSystem.NUM_PARTICLES, GetAssetFullPath(L"compute.hlsl"));

}

void D3D12Renderer::DispatchCompute()
{
    // reset
    ThrowIfFailed(m_computeAllocator->Reset());
    ThrowIfFailed(m_computeCommandList->Reset(m_computeAllocator.Get(), m_computeTestPipeline.Get()));

    // set states
    m_computeCommandList->SetComputeRootSignature(m_computeTestRootSignature.Get());
    m_computeCommandList->SetPipelineState(m_computeTestPipeline.Get());
    m_computeCommandList->SetComputeRootUnorderedAccessView(0, m_computeTestBuffer->GetGPUVirtualAddress());

    // dispatch
    m_computeCommandList->Dispatch((m_particleSystem.NUM_PARTICLES + 63) / 64, 1, 1);

    ThrowIfFailed(m_computeCommandList->Close());

    // execute compute on the compute command queue
    ID3D12CommandList* computeLists[] = { m_computeCommandList.Get() };
    m_computeCommandQueue->ExecuteCommandLists(1, computeLists);

    // waitforgpu but with the compute pipeline
    m_computeFenceValue++;
    ThrowIfFailed(m_computeCommandQueue->Signal(m_computeFence.Get(), m_computeFenceValue));
    ThrowIfFailed(m_computeFence->SetEventOnCompletion(m_computeFenceValue, m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
}

void D3D12Renderer::PopulateCommandList()
{
	// reset the command allocator.
	// can only work if the command lists have finished execution on gpu.
	ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

	// also need to reset command list
	ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

	// set states
	m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
	m_commandList->RSSetViewports(1, &m_viewport);
	m_commandList->RSSetScissorRects(1, &m_scissorRect);
	m_commandList->SetGraphicsRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

	// create resource barriers and stuff for back buffer
	CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
		m_renderTargets[m_frameIndex].Get(),
		D3D12_RESOURCE_STATE_PRESENT,
		D3D12_RESOURCE_STATE_RENDER_TARGET);

	m_commandList->ResourceBarrier(1, &barrier);

	// get handles
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(m_dsvHeap->GetCPUDescriptorHandleForHeapStart());

	// ----- commands start here -----
	const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
	m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	m_commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	// bind render targets
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

	// indexed draw
	m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	D3D12_VERTEX_BUFFER_VIEW views[2] = { m_vertexBufferView, m_particleSystem.m_instancer.m_instanceBufferView };
	m_commandList->IASetVertexBuffers(0, 2, views);
	m_commandList->IASetIndexBuffer(&m_indexBufferView);
	m_particleSystem.m_instancer.TransitionToVertexBuffer(m_commandList.Get()); // start working with the particles
	m_commandList->DrawIndexedInstanced(m_particleSystem.m_instancer.m_sphereIndexCount, m_particleSystem.NUM_PARTICLES, 0, 0, 0);

	// present back buffer
	CD3DX12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
		m_renderTargets[m_frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT);
	m_commandList->ResourceBarrier(1, &barrier2);

	ThrowIfFailed(m_commandList->Close());
}

void D3D12Renderer::MoveToNextFrame()
{
	 // Schedule a Signal command in the queue.
    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

    // Update the frame index.
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // If the next frame is not ready to be rendered yet, wait until it is ready.
    if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
    {
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
        WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
    }

    // Set the fence value for the next frame.
    m_fenceValues[m_frameIndex] = currentFenceValue + 1;
}

void D3D12Renderer::WaitForGPU()
{
	ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

    // Wait until the fence has been processed.
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

    // Increment the fence value for the current frame.
    m_fenceValues[m_frameIndex]++;
}

void D3D12Renderer::ReadbackCompute()
{
    // need a fresh command list to record the copy
    ThrowIfFailed(m_computeAllocator->Reset());
    ThrowIfFailed(m_computeCommandList->Reset(m_computeAllocator.Get(), m_computeTestPipeline.Get()));

    // transition compute buffer from UAV -> copy source
    CD3DX12_RESOURCE_BARRIER toCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(
        m_computeTestBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    m_computeCommandList->ResourceBarrier(1, &toCopySrc);

    m_computeCommandList->CopyResource(m_computeReadbackBuffer.Get(), m_computeTestBuffer.Get());

    // transition back so next frame's dispatch can write to it again
    CD3DX12_RESOURCE_BARRIER toUAV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_computeTestBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    m_computeCommandList->ResourceBarrier(1, &toUAV);

    ThrowIfFailed(m_computeCommandList->Close());

    ID3D12CommandList* lists[] = { m_computeCommandList.Get() };
    m_computeCommandQueue->ExecuteCommandLists(1, lists);

    // wait for copy to finish
    m_computeFenceValue++;
    ThrowIfFailed(m_computeCommandQueue->Signal(m_computeFence.Get(), m_computeFenceValue));
    ThrowIfFailed(m_computeFence->SetEventOnCompletion(m_computeFenceValue, m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

    // map and print
    UINT byteSize = m_particleSystem.NUM_PARTICLES * sizeof(XMFLOAT4);
    XMFLOAT4* pData = nullptr;
    CD3DX12_RANGE readRange(0, byteSize);
    ThrowIfFailed(m_computeReadbackBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pData)));

    for (int i = 0; i < 8; i++)
    {
        char buf[64];
        sprintf_s(buf, "particle[%d] = (%.1f, %.1f, %.1f, %.1f)\n",
            i, pData[i].x, pData[i].y, pData[i].z, pData[i].w);
        OutputDebugStringA(buf);
    }

    m_computeReadbackBuffer->Unmap(0, nullptr);
}