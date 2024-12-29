cmake_minimum_required(VERSION 3.18)

project(ScudaProject LANGUAGES CXX CUDA)

set(MIN_CUDA_VERSION 12.0)
set(MAX_CUDA_VERSION 12.7)

find_package(CUDAToolkit REQUIRED)

if (NOT CUDAToolkit_VERSION)
    message(FATAL_ERROR "CUDA Toolkit is required but not found.")
endif()

if (CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION OR CUDAToolkit_VERSION VERSION_GREATER MAX_CUDA_VERSION)
    message(FATAL_ERROR "CUDA Toolkit version must be >= ${MIN_CUDA_VERSION} and <= ${MAX_CUDA_VERSION}. Found: ${CUDAToolkit_VERSION}")
endif()

message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    ${CUDAToolkit_INCLUDE_DIRS}
)

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

set(CLIENT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/manual_client.h
)

set(SERVER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/manual_server.h
)

set(CLIENT_OUTPUT scuda_${CUDAToolkit_VERSION_MAJOR}_${CUDAToolkit_VERSION_MINOR})
set(SERVER_OUTPUT server_${CUDAToolkit_VERSION_MAJOR}_${CUDAToolkit_VERSION_MINOR}.so)

find_library(CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS /usr/local/cuda/lib64)
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS /usr/local/cuda/lib64)
find_library(CUDART_LIBRARY cudart HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS /usr/local/cuda/lib64)
find_library(NVML_LIBRARY nvidia-ml HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS /usr/local/cuda/lib64)

if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS library not found. Ensure CUDA is installed and the library path is correct.")
endif()

if (NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found. Ensure CUDA is installed and the library path is correct.")
endif()

if (NOT CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA Runtime library (libcudart.so) not found. Ensure CUDA is installed and the library path is correct.")
endif()

if (NOT NVML_LIBRARY)
    message(FATAL_ERROR "NVML library not found. Ensure CUDA is installed and the library path is correct.")
endif()

message(STATUS "Found cuBLAS library: ${CUBLAS_LIBRARY}")
message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")
message(STATUS "Found CUDA Runtime library: ${CUDART_LIBRARY}")
message(STATUS "Found NVML library: ${NVML_LIBRARY}")

add_library(${CLIENT_OUTPUT} SHARED ${CLIENT_SOURCES} ${CLIENT_HEADERS})
target_include_directories(${CLIENT_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${CLIENT_OUTPUT} PRIVATE ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} stdc++)
set_target_properties(${CLIENT_OUTPUT} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(${SERVER_OUTPUT} ${SERVER_SOURCES} ${SERVER_HEADERS})
target_include_directories(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_LIBRARIES} cuda)
target_link_libraries(${SERVER_OUTPUT} PRIVATE ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUDNN_LIBRARY} ${NVML_LIBRARY})

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set_source_files_properties(
    ${SERVER_SOURCES}
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(${SERVER_OUTPUT} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options=-fPIC,-g,-Wno-deprecated-declarations>
)

message(STATUS "Building client output: ${CLIENT_OUTPUT}")
message(STATUS "Building server output: ${SERVER_OUTPUT}")
