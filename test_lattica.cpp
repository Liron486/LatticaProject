#include <gtest/gtest.h>
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

// Test fixture for compare kernel
class CompareKernelTest : public ::testing::Test {
protected:
    torch::Tensor a, b, result;

    void SetUp() override {
        // Allocate tensors
        a = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        b = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        result = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);
    }
};

// Test Case 1: Equal values
TEST_F(CompareKernelTest, EqualValues) {
    a.fill_(1);
    b.fill_(1);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for equal values";
    }
}

// Test Case 2: a > b
TEST_F(CompareKernelTest, AGreaterThanB) {
    a.fill_(2);
    b.fill_(1);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for a > b";
    }
}

// Test Case 3: b > a
TEST_F(CompareKernelTest, BGreaterThanA) {
    a.fill_(1);
    b.fill_(2);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 0) << "Failed for b > a";
    }
}

// Test Case 4: Edge case - All zeros
TEST_F(CompareKernelTest, AllZeros) {
    a.zero_();
    b.zero_();

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for all zeros";
    }
}

// Test Case 5: Edge case - Maximum values
TEST_F(CompareKernelTest, MaxValues) {
    uint32_t max_value = (1u << 32) - 1;
    a.fill_(max_value);
    b.fill_(max_value);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for max values";
    }
}

// Test suite for add kernel
class LaunchAddKernelTest : public ::testing::Test {
protected:
    torch::Tensor a, b, add_result, carry_result;

    void SetUp() override {
        // Allocate tensors
        a = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        b = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        add_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        carry_result = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);
    }
};

// Test Case 1: Adding two tensors with all elements as 0
TEST_F(LaunchAddKernelTest, AddZeros) {
    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 2: Adding two tensors with identical values
TEST_F(LaunchAddKernelTest, AddIdenticalValues) {
    a.fill_(1);
    b.fill_(1);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::full({ n, k }, 2, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 3: Adding two tensors with one tensor all zeros
TEST_F(LaunchAddKernelTest, AddWithZeros) {
    b.fill_(42);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::full({ n, k }, 42, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 4: Carry propagation across segments
TEST_F(LaunchAddKernelTest, CarryPropagationWithinTensor) {
    a.fill_(UINT32_MAX);
    b.fill_(1);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::full({ n }, 1, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 5: Carry spanning entire tensor
TEST_F(LaunchAddKernelTest, CarrySpansEntireTensor) {
    for (int i = 0; i < k; ++i) {
        a.select(1, i).fill_(UINT32_MAX);
    }
    b.select(1, 0).fill_(1);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::full({ n }, 1, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 6: Maximum values in all segments
TEST_F(LaunchAddKernelTest, AddMaxValues) {
    a.fill_(UINT32_MAX);
    b.fill_(UINT32_MAX);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::full({ n, k }, UINT32_MAX - 1, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::full({ n }, 1, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 7: Alternating large and small values
TEST_F(LaunchAddKernelTest, AlternatingValues) {
    for (int i = 0; i < k; ++i) {
        a.select(1, i).fill_(i % 2 == 0 ? UINT32_MAX : 0);
        b.select(1, i).fill_(i % 2 == 0 ? 1 : UINT32_MAX);
    }

    launchAddKernel(a, b, add_result, carry_result);

    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::full({ n }, 1, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test Case 8: Randomized Testing
TEST_F(LaunchAddKernelTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    auto random_tensor = [&]() {
        auto t = torch::empty({ n, k }, torch::kUInt32).to(torch::kCUDA);
        auto* data_ptr = t.data_ptr<uint32_t>();
        for (int i = 0; i < n * k; ++i) {
            data_ptr[i] = dist(gen);
        }
        return t;
        };

    a = random_tensor();
    b = random_tensor();

    launchAddKernel(a, b, add_result, carry_result);

    // Validate correctness on CPU for smaller batches
    auto a_cpu = a.to(torch::kCPU);
    auto b_cpu = b.to(torch::kCPU);
    auto result_cpu = add_result.to(torch::kCPU);
    auto carry_cpu = carry_result.to(torch::kCPU);

    uint32_t* a_data = a_cpu.data_ptr<uint32_t>();
    uint32_t* b_data = b_cpu.data_ptr<uint32_t>();
    uint32_t* result_data = result_cpu.data_ptr<uint32_t>();
    int* carry_data = carry_cpu.data_ptr<int>();

    for (int i = 0; i < n; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < k; ++j) {
            uint64_t sum = (uint64_t)a_data[i * k + j] +
                (uint64_t)b_data[i * k + j] + carry;
            uint32_t expected_value = sum & UINT32_MAX;
            carry = sum >> 32;

            ASSERT_EQ(result_data[i * k + j], expected_value)
                << "Mismatch at [" << i << "][" << j << "]";
        }
        ASSERT_EQ(carry_data[i], carry) << "Mismatch in carry for batch " << i;
    }
}
