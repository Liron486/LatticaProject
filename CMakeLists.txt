cmake_minimum_required(VERSION 3.18)
project(Lattica LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the path to libtorch
set(CMAKE_PREFIX_PATH "C:\\Users\\liron\\libtorch")

# Set the policy to allow FindCUDA (if necessary)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# Find CUDA (required before finding Torch to resolve dependencies)
find_package(CUDA REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)

# Add the executable
add_executable(Lattica torch_cuda_app.cu)

# Link Torch libraries and CUDA::nvToolsExt
target_link_libraries(Lattica PRIVATE "${TORCH_LIBRARIES}")

# Include Torch headers scoped to the target
target_include_directories(Lattica PRIVATE ${TORCH_INCLUDE_DIRS})

# Optional: Enable CUDA separable compilation if needed
set_property(TARGET Lattica PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Optional: Print out Torch and CUDA details for debugging
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA::nvToolsExt path: $<TARGET_PROPERTY:CUDA::nvToolsExt,INTERFACE_LINK_LIBRARIES>")
