cmake_minimum_required(VERSION 3.20)
project("Phthalo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Hot Reload for MSVC compilers if supported.
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "src/*.h")

# Add source to this project's executable.
add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS})

# precompile headers
target_precompile_headers(${PROJECT_NAME} PRIVATE src/stdafx.h)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
    dxguid.lib
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    UNICODE
    _UNICODE
)

# shader compilation
# Copy shader file to build directory so D3DCompileFromFile can find it
file(GLOB HLSL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.hlsl")

foreach(HLSL_FILE ${HLSL_FILES})
    get_filename_component(HLSL_NAME ${HLSL_FILE} NAME)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${HLSL_FILE}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/${HLSL_NAME}
        COMMENT "Copying ${HLSL_NAME} to output directory"
    )
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES
    WIN32_EXECUTABLE TRUE
)
