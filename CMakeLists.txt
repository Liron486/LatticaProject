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

# Option to include or exclude tests
option(ENABLE_TESTS "Include Google Test-based unit tests in the build" ON)

# Add the executable with conditional inclusion of test files
add_executable(Lattica
    torch_cuda_app.cu
    $<$<BOOL:${ENABLE_TESTS}>:test_lattica.cpp> # Add the test file only if ENABLE_TESTS is ON
)

# Link Torch libraries and CUDA::nvToolsExt
target_link_libraries(Lattica PRIVATE "${TORCH_LIBRARIES}")

# Include Torch headers scoped to the target
target_include_directories(Lattica PRIVATE ${TORCH_INCLUDE_DIRS})

# Optional: Enable CUDA separable compilation if needed
set_property(TARGET Lattica PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include Google Test if ENABLE_TESTS is ON
if(ENABLE_TESTS)
    add_subdirectory(googletest)
    target_link_libraries(Lattica PRIVATE gtest gtest_main)
    target_include_directories(Lattica PRIVATE googletest/include)
    target_compile_definitions(Lattica PRIVATE ENABLE_TESTS=1)
endif()

# Display configuration status
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA::nvToolsExt path: $<TARGET_PROPERTY:CUDA::nvToolsExt,INTERFACE_LINK_LIBRARIES>")
message(STATUS "ENABLE_TESTS is ${ENABLE_TESTS}")
