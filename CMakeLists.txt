cmake_minimum_required(VERSION 3.18)

project(ScudaProject LANGUAGES CXX CUDA)

set(MIN_CUDA_VERSION 12.0)
set(MAX_CUDA_VERSION 12.4)

find_package(CUDAToolkit REQUIRED)

if (NOT CUDAToolkit_VERSION)
    message(FATAL_ERROR "CUDA Toolkit is required but not found.")
endif ()

if (CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION OR CUDAToolkit_VERSION VERSION_GREATER MAX_CUDA_VERSION)
    message(FATAL_ERROR "CUDA Toolkit version must be >= ${MIN_CUDA_VERSION} and <= ${MAX_CUDA_VERSION}. Found: ${CUDAToolkit_VERSION}")
endif ()

message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CLIENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/manual_client.cpp
)

set(SERVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/server.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/manual_server.cpp
)

set(CLIENT_OUTPUT libscuda_${CUDAToolkit_VERSION_MAJOR}_${CUDAToolkit_VERSION_MINOR}.so)
set(SERVER_OUTPUT server_${CUDAToolkit_VERSION_MAJOR}_${CUDAToolkit_VERSION_MINOR}.so)

set(CUDA_LIBRARY_PATH /usr/local/cuda/lib64)
set(CUBLAS_LIBRARY ${CUDA_LIBRARY_PATH}/libcublas.so)
set(CUDNN_LIBRARY ${CUDA_LIBRARY_PATH}/libcudnn.so)
set(CUDART_LIBRARY ${CUDA_LIBRARY_PATH}/libcudart.so)
set(NVML_LIBRARY ${CUDA_LIBRARY_PATH}/libnvidia-ml.so)

add_library(${CLIENT_OUTPUT} SHARED ${CLIENT_SOURCES})
target_include_directories(${CLIENT_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${CLIENT_OUTPUT} PRIVATE ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} stdc++)
set_target_properties(${CLIENT_OUTPUT} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(${SERVER_OUTPUT} ${SERVER_SOURCES})
target_include_directories(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${SERVER_OUTPUT} PRIVATE ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUDNN_LIBRARY} ${NVML_LIBRARY})

set_source_files_properties(${SERVER_SOURCES} PROPERTIES LANGUAGE CUDA)

target_compile_options(${SERVER_OUTPUT} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC -g -Wno-deprecated-declarations>
)

message(STATUS "Building client output: ${CLIENT_OUTPUT}")
message(STATUS "Building server output: ${SERVER_OUTPUT}")
