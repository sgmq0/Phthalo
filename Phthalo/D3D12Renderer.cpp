
#include "stdafx.h"
#include "D3D12Renderer.h"

D3D12Renderer::D3D12Renderer(UINT width, UINT height) :
	DXSample(width, height, L"Renderer"),
	m_frameIndex(0),
	m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)),
	m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
	m_rtvDescriptorSize(0)
{
}

void D3D12Renderer::OnInit()
{
	LoadPipeline();
	LoadAssets();
}

void D3D12Renderer::OnUpdate()
{
}

void D3D12Renderer::OnRender()
{
}

void D3D12Renderer::OnDestroy()
{
}

void D3D12Renderer::LoadPipeline()
{

}

void D3D12Renderer::LoadAssets()
{

}

void D3D12Renderer::PopulateCommandList()
{
}

void D3D12Renderer::WaitForPreviousFrame()
{
}
