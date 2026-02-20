
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
	m_camera(XMFLOAT3(0.0f, 0.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f),
	m_sphere(SphereMesh(0.1f))
{
}

void D3D12Renderer::OnInit()
{
	LoadPipeline();
	LoadParticles();
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

	UpdateParticles(dt);

	memcpy(m_pInstanceDataBegin, m_instances.data(), sizeof(InstanceData) * NUM_PARTICLES); // particles!!!
}

void D3D12Renderer::OnRender()
{
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


void D3D12Renderer::LoadParticles()
{
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
    m_instances.resize(NUM_PARTICLES);
}

float Poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 >= h2) return 0.0f;
    float x = h2 - r2;
    return (315.0f / (64.0f * XM_PI * pow(h, 9))) * x * x * x;
}

XMFLOAT3 SpikyGradient(XMFLOAT3 ri, XMFLOAT3 rj, float h) {
    float dx = ri.x - rj.x;
    float dy = ri.y - rj.y;
    float dz = ri.z - rj.z;
    float r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-6f || r >= h) return {0, 0, 0};
    float coeff = -45.0f / (XM_PI * pow(h, 6)) * (h - r) * (h - r) / r;
    return { coeff * dx, coeff * dy, coeff * dz };
}

void D3D12Renderer::SolveConstraints(float dist, float distSquared) {
    const float rho0    = 10.0f;
    const float epsilon = 1000.0f;
    const int   iters   = 3;

    for (int iter = 0; iter < iters; iter++) {

        // compute lambda for each particle
        for (int i = 0; i < NUM_PARTICLES; i++) {
            // estimate density via Poly6
            //float rho = Poly6(0.0f, dist);  // self contribution
			float rho = rho0;

            for (int j : m_particles[i].neighbors) {
                float dx = m_particles[i].predictedPosition.x - m_particles[j].predictedPosition.x;
                float dy = m_particles[i].predictedPosition.y - m_particles[j].predictedPosition.y;
                float dz = m_particles[i].predictedPosition.z - m_particles[j].predictedPosition.z;
                rho += Poly6(dx*dx + dy*dy + dz*dz, dist);
            }

            // density constraint C_i = rho/rho0 - 1
            float C = rho / rho0 - 1.0f;

            // sum of squared gradients for denominator
            float sumGrad2 = 0.0f;
            for (int n = 0; n < m_particles[i].neighbors.size(); n++) {
                int j = m_particles[i].neighbors[n];
                XMFLOAT3 grad = SpikyGradient(m_particles[i].predictedPosition, m_particles[j].predictedPosition, dist);
                float g = (grad.x*grad.x + grad.y*grad.y + grad.z*grad.z) / rho0;
                sumGrad2 += g;
            }

            m_particles[i].lambda = -C / (sumGrad2 + epsilon);
        }

        // compute and apply delta p
        std::vector<XMFLOAT3> deltas(NUM_PARTICLES, {0.0f, 0.0f, 0.0f});

		for (int i = 0; i < NUM_PARTICLES; i++) {
			for (int j : m_particles[i].neighbors) {
				float w = m_particles[i].lambda + m_particles[j].lambda;
				XMFLOAT3 grad = SpikyGradient(m_particles[i].predictedPosition, m_particles[j].predictedPosition, dist);
				deltas[i].x += w * grad.x / rho0;
				deltas[i].y += w * grad.y / rho0;
				deltas[i].z += w * grad.z / rho0;
			}
		}

		for (int i = 0; i < NUM_PARTICLES; i++) {
			m_particles[i].predictedPosition.x += deltas[i].x;
			m_particles[i].predictedPosition.y += deltas[i].y;
			m_particles[i].predictedPosition.z += deltas[i].z;
		}
    }
}

void D3D12Renderer::UpdateParticles(float dt)
{
	if (dt <= 0.0f || dt > 0.1f) return;

	const float dist = 1.0f;
	const float distSquared = dist * dist;

	for (int i = 0; i < NUM_PARTICLES; i++) {

		// apply forces
		m_particles[i].velocity.y += -9.8f * dt;

		// predict position
		m_particles[i].predictedPosition.x = m_particles[i].position.x + dt * m_particles[i].velocity.x;
		m_particles[i].predictedPosition.y = m_particles[i].position.y + dt * m_particles[i].velocity.y;
		m_particles[i].predictedPosition.z = m_particles[i].position.z + dt * m_particles[i].velocity.z;
    }

	// find neighbors
	for (int i = 0; i < NUM_PARTICLES; i++) {
		m_particles[i].neighbors.clear();
	}

	for (int i = 0; i < NUM_PARTICLES; i++) {
		for (int j = i + 1; j < NUM_PARTICLES; j++) {
			float dx = m_particles[i].predictedPosition.x - m_particles[j].predictedPosition.x;
			float dy = m_particles[i].predictedPosition.y - m_particles[j].predictedPosition.y;
			float dz = m_particles[i].predictedPosition.z - m_particles[j].predictedPosition.z;
			float dist2 = dx*dx + dy*dy + dz*dz;

			if (dist2 < distSquared) {
				m_particles[i].neighbors.push_back(j);
				m_particles[j].neighbors.push_back(i);
			}
		}
	}

	// calculations for all particles
	SolveConstraints(dist, distSquared);

	// update positions and velocity
	for (int i = 0; i < NUM_PARTICLES; i++) {
		if (m_particles[i].predictedPosition.y < 0.0f) {
			m_particles[i].predictedPosition.y = 0.0f;
		}

		const float boxMin = -2.0f;
		const float boxMax =  2.0f;

		if (m_particles[i].predictedPosition.x < boxMin) { m_particles[i].predictedPosition.x = boxMin; }
		if (m_particles[i].predictedPosition.x > boxMax) { m_particles[i].predictedPosition.x = boxMax; }
		if (m_particles[i].predictedPosition.y < boxMin) { m_particles[i].predictedPosition.y = boxMin; }
		if (m_particles[i].predictedPosition.z < boxMin) { m_particles[i].predictedPosition.z = boxMin; }
		if (m_particles[i].predictedPosition.z > boxMax) { m_particles[i].predictedPosition.z = boxMax; }

		m_particles[i].velocity.x = (m_particles[i].predictedPosition.x - m_particles[i].position.x) / dt;
		m_particles[i].velocity.y = (m_particles[i].predictedPosition.y - m_particles[i].position.y) / dt;
		m_particles[i].velocity.z = (m_particles[i].predictedPosition.z - m_particles[i].position.z) / dt;
		m_particles[i].position = m_particles[i].predictedPosition;

		if (m_particles[i].position.y < 0.0f) {
			m_particles[i].velocity.y *= -0.3f; // dampen bounce
		}
	}

	// push to instances vector
	for (int i = 0; i < NUM_PARTICLES; i++) {
        XMMATRIX mat = XMMatrixTranslation(
            m_particles[i].position.x,
            m_particles[i].position.y,
            m_particles[i].position.z
        );

        XMStoreFloat4x4(&m_instances[i].worldMatrix, mat);
    }

	// for (int i = 0; i < NUM_PARTICLES; i++) {
	// 	m_particles[i].velocity.y += -9.8f * dt;
    //     m_particles[i].position.x += m_particles[i].velocity.x * dt;
    //     m_particles[i].position.y += m_particles[i].velocity.y * dt;
    //     m_particles[i].position.z += m_particles[i].velocity.z * dt;

	// 	if (m_particles[i].position.y < 0) {
	// 		m_particles[i].velocity.y *= -1;
	// 	}

    //     // write to instance buffer to actually draw the points
    //     XMMATRIX mat = XMMatrixTranslation(
    //         m_particles[i].position.x,
    //         m_particles[i].position.y,
    //         m_particles[i].position.z
    //     );

    //     XMStoreFloat4x4(&m_instances[i].worldMatrix, mat);
    // }
}

void D3D12Renderer::LoadAssets()
{
	// load sphere mesh
	m_sphere.LoadMesh();
	m_sphereIndexCount = m_sphere.sphereIndices.size();

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
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);				// no transparency by default
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

	// ---------- create the vertex buffer ----------
	{
		const UINT vertexBufferSize = m_sphere.sphereVertices.size() * sizeof(SphereMesh::Vertex);

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
		CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
		ThrowIfFailed(m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
		memcpy(pVertexDataBegin, m_sphere.sphereVertices.data(), vertexBufferSize);
		m_vertexBuffer->Unmap(0, nullptr);

		// init vertex buffer view
		m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
		m_vertexBufferView.StrideInBytes = sizeof(SphereMesh::Vertex);
		m_vertexBufferView.SizeInBytes = vertexBufferSize;
	}

	// ---------- create the index buffer ----------
	{
		const UINT indexBufferSize = m_sphere.sphereIndices.size() * sizeof(UINT32);

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
		memcpy(pIndexDataBegin, m_sphere.sphereIndices.data(), indexBufferSize);
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

		// map it once and leave it mapped — safe for upload heaps
		CD3DX12_RANGE readRange(0, 0);
		ThrowIfFailed(m_constantBuffer->Map(0, &readRange,
			reinterpret_cast<void**>(&m_pCbvDataBegin)));
	}

	// ---------- create the instancing buffer ----------
	{
		const UINT instanceBufferSize = NUM_PARTICLES * sizeof(InstanceData);

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(instanceBufferSize);
		ThrowIfFailed(m_device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_instanceBuffer)));

		// copy instance data to instance buffer
		CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
		ThrowIfFailed(m_instanceBuffer->Map(0, &readRange, 
			reinterpret_cast<void**>(&m_pInstanceDataBegin)));

		// init instance buffer view
		m_instanceBufferView.BufferLocation = m_instanceBuffer->GetGPUVirtualAddress();
		m_instanceBufferView.StrideInBytes = sizeof(InstanceData);
		m_instanceBufferView.SizeInBytes = instanceBufferSize;
	}

	// ---------- synchronization objects, fences and such ----------
	// create fence and wait for the gpu.
	ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
	m_fenceValues[m_frameIndex]++;

	// create fence event handle for frame synchronization.
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}

	// wait for setup to complete, in this case
	WaitForGPU();
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
	D3D12_VERTEX_BUFFER_VIEW views[2] = { m_vertexBufferView, m_instanceBufferView };
	m_commandList->IASetVertexBuffers(0, 2, views);
	m_commandList->IASetIndexBuffer(&m_indexBufferView);
	m_commandList->DrawIndexedInstanced(m_sphereIndexCount, NUM_PARTICLES, 0, 0, 0);

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
