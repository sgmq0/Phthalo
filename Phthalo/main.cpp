#include "stdafx.h"

#include "D3D12HelloTriangle.h"
#include "D3D12Renderer.h"

_Use_decl_annotations_
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) 
{
	D3D12Renderer renderer(1280, 720);
	return Win32Application::Run(&renderer, hInstance, nCmdShow);
}