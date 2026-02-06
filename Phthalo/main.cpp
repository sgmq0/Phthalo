#include "stdafx.h"

#include "D3D12HelloTriangle.h"

_Use_decl_annotations_
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {

	// this is a DXSample that is a container class for everything we make.
	// eventually: create a D3D12 renderer class
	D3D12HelloTriangle sample(1280, 720, L"D3D12 Hello Triangle");

	return Win32Application::Run(&sample, hInstance, nCmdShow);
}