#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/cuda.h>

#include "torch_cuda_app.hpp"

// A test to check if CUDA is available
TEST(CUDATest, CheckCudaAvailability) {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        ASSERT_TRUE(true);
    }
    else {
        std::cerr << "CUDA is not available!" << std::endl;
        ASSERT_TRUE(false) << "CUDA should be available for this application.";
    }
}

// Test suite for compare kernel
TEST(CompareKernelTest, BasicCases)
{
    // Set up tensors
    torch::Device device(torch::kCUDA);
    torch::Tensor a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor result = torch::zeros({ n }, torch::kInt32).to(device);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test 1: Equal values
    a.fill_(1);
    b.fill_(1);
    launchCompareKernel(a, b, result);
    cudaStreamSynchronize(stream);
    auto result_cpu = result.to(torch::kCPU);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for equal values";
    }

    // Test 2: a > b
    a.fill_(2);
    b.fill_(1);
    launchCompareKernel(a, b, result, stream);
    cudaStreamSynchronize(stream);
    result_cpu = result.to(torch::kCPU);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for a > b";
    }

    // Test 3: b > a
    a.fill_(1);
    b.fill_(2);
    launchCompareKernel(a, b, result, stream);
    cudaStreamSynchronize(stream);
    result_cpu = result.to(torch::kCPU);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 0) << "Failed for b > a";
    }

    // Test 4: Edge case - All zeros
    a.zero_();
    b.zero_();
    launchCompareKernel(a, b, result, stream);
    cudaStreamSynchronize(stream);
    result_cpu = result.to(torch::kCPU);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for all zeros";
    }

    // Test 5: Edge case - Maximum values
    uint32_t max_value = (1u << 32) - 1;
    a.fill_(max_value);
    b.fill_(max_value);
    launchCompareKernel(a, b, result, stream);
    cudaStreamSynchronize(stream);
    result_cpu = result.to(torch::kCPU);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for max values";
    }

    // Cleanup
    cudaStreamDestroy(stream);
}

