#include "stdafx.h"

#include "Win32Application.h"

_Use_decl_annotations_
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	return Win32Application::Run(nullptr, hInstance, nCmdShow);
}