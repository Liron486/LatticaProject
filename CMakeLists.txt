cmake_minimum_required(VERSION 3.18)
project(Lattica LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the path to libtorch (adjust this to your libtorch installation directory)
set(CMAKE_PREFIX_PATH "C:\\Users\\liron\\libtorch")

# Find PyTorch
find_package(Torch REQUIRED)

# Option to include or exclude tests
option(ENABLE_TESTS "Include Google Test-based unit tests in the build" ON)

# Add the executable with conditional inclusion of test files
add_executable(Lattica
    torch_cuda_app.cu
    $<$<BOOL:${ENABLE_TESTS}>:test_lattica.cpp> # Add the test file only if ENABLE_TESTS is ON
)

# Include the header file
target_include_directories(Lattica PRIVATE ${CMAKE_SOURCE_DIR})

# Link Torch libraries and CUDA libraries
target_link_libraries(Lattica PRIVATE "${TORCH_LIBRARIES}" cuda cudart)

# Optional: Enable CUDA separable compilation if needed
set_property(TARGET Lattica PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include Google Test if ENABLE_TESTS is ON
if(ENABLE_TESTS)
    # Add GoogleTest as a subdirectory (assumes googletest source is in the project folder)
    add_subdirectory(googletest)

    # Link Google Test libraries to the target
    target_link_libraries(Lattica PRIVATE gtest gtest_main)

    # Include Google Test headers
    target_include_directories(Lattica PRIVATE googletest/include)

    # Add a definition for conditional compilation
    target_compile_definitions(Lattica PRIVATE ENABLE_TESTS=1)
endif()

# Display configuration status
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA library path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "ENABLE_TESTS is ${ENABLE_TESTS}")
