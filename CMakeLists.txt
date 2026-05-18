cmake_minimum_required(VERSION 3.18)

project(ScudaProject LANGUAGES CXX CUDA)

set(MIN_CUDA_VERSION 11.0)

find_package(CUDAToolkit REQUIRED)

if (NOT CUDAToolkit_VERSION)
    message(FATAL_ERROR "CUDA Toolkit is required but not found.")
endif()

if (CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION)
    message(FATAL_ERROR "CUDA Toolkit version must be >= ${MIN_CUDA_VERSION}. Found: ${CUDAToolkit_VERSION}")
endif()

message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(CLIENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_client.cpp
)

set(SERVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/manual_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.cpp
)

set(CLIENT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.h
)

set(SERVER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/manual_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
)

set(CLIENT_OUTPUT scuda)
set(SERVER_OUTPUT scuda_driver_server)

add_library(${CLIENT_OUTPUT} SHARED ${CLIENT_SOURCES} ${CLIENT_HEADERS})
target_include_directories(${CLIENT_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${CLIENT_OUTPUT} PRIVATE stdc++)
target_link_options(${CLIENT_OUTPUT} PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/scuda.exports"
)
set_target_properties(${CLIENT_OUTPUT} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME scuda
    VERSION 1
    SOVERSION 1
)

add_custom_command(TARGET ${CLIENT_OUTPUT} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -f
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libcuda.so.1
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libcuda.so
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libnvidia-ml.so.1
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libnvidia-ml.so
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:${CLIENT_OUTPUT}>
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libcuda.so.1
    COMMAND ${CMAKE_COMMAND} -E create_symlink libcuda.so.1
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libcuda.so
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:${CLIENT_OUTPUT}>
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libnvidia-ml.so.1
    COMMAND ${CMAKE_COMMAND} -E create_symlink libnvidia-ml.so.1
            $<TARGET_FILE_DIR:${CLIENT_OUTPUT}>/libnvidia-ml.so
)

add_custom_target(scuda_driver DEPENDS ${CLIENT_OUTPUT})
add_custom_target(scuda_nvml DEPENDS ${CLIENT_OUTPUT})

add_executable(${SERVER_OUTPUT} ${SERVER_SOURCES} ${SERVER_HEADERS})
target_include_directories(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_LIBRARIES} cuda dl)

set_source_files_properties(
    ${SERVER_SOURCES}
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(${SERVER_OUTPUT} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options=-fPIC,-g,-Wno-deprecated-declarations>
)

target_compile_options(${CLIENT_OUTPUT} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -Wno-deprecated-declarations>
)

message(STATUS "Building client output: ${CLIENT_OUTPUT}")
message(STATUS "Building server output: ${SERVER_OUTPUT}")
