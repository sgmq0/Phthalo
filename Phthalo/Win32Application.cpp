#include "stdafx.h"
#include "Win32Application.h"

HWND Win32Application::m_hwnd = nullptr;

int Win32Application::Run(DXSample* pSample, HINSTANCE hInstance, int nCmdShow)
{
	const int width = 800;
	const int height = 800;

	// simple window class
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = L"DirectX12Triangle";
	RegisterClass(&wc);

	HWND hwnd = CreateWindow(
		wc.lpszClassName, 
		L"DirectX 12 Triangle", 
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 
		CW_USEDEFAULT, 
		width, 
		height, 
		nullptr, 
		nullptr, 
		hInstance, 
		nullptr);

	ShowWindow(hwnd, nCmdShow);

	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		// Process any messages in the queue.
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return 1;

}

LRESULT Win32Application::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message) {
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}