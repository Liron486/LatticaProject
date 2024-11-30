#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/cuda.h>

// A simple test to verify Google Test setup
TEST(SimpleTest, BasicAssertion) {
    // Test passes if 1 is equal to 1
    ASSERT_EQ(1, 1) << "1 should be equal to 1.";
}

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

// A test to check if PyTorch tensors can be created on CUDA
TEST(TensorTest, CreateCudaTensor) {
    if (torch::cuda::is_available()) {
        // Create a tensor on CUDA
        auto tensor = torch::rand({ 10, 10 }, torch::device(torch::kCUDA));
        ASSERT_EQ(tensor.device().type(), torch::kCUDA) << "Tensor should be on CUDA device.";
        ASSERT_EQ(tensor.sizes(), torch::IntArrayRef({ 10, 10 })) << "Tensor size should be 10x10.";
    }
    else {
        GTEST_SKIP() << "CUDA is not available. Skipping tensor creation test.";
    }
}
