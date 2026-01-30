cmake_minimum_required(VERSION 3.21)
project(DX12App LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows only
if (NOT WIN32)
    message(FATAL_ERROR "DirectX 12 requires Windows")
endif()

add_executable(DX12App
    src/main.cpp
)

# Enable Unicode + warnings
target_compile_definitions(DX12App PRIVATE UNICODE _UNICODE)
target_compile_options(DX12App PRIVATE /W4)

# DirectX 12 libraries
target_link_libraries(DX12App PRIVATE
    d3d12
    dxgi
    dxguid
)

# Optional: make VS debugger start in project folder
set_property(TARGET DX12App PROPERTY VS_DEBUGGER_WORKING_DIRECTORY
    "${CMAKE_SOURCE_DIR}"
)
