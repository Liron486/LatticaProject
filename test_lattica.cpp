#include <gtest/gtest.h>
#include "torch_cuda_app.hpp"

// A test to check if CUDA is available
TEST(CUDATest, CheckCudaAvailability) {
    if (torch::cuda::is_available()) {
        ASSERT_TRUE(true);
    }
    else {
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

// Compare Test Case 1: Equal values
TEST_F(CompareKernelTest, EqualValues) {
    a.fill_(1);
    b.fill_(1);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for equal values";
    }
}

// Compare Test Case 2: a > b
TEST_F(CompareKernelTest, AGreaterThanB) {
    a.fill_(2);
    b.fill_(1);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for a > b";
    }
}

// Compare Test Case 3: b > a
TEST_F(CompareKernelTest, BGreaterThanA) {
    a.fill_(1);
    b.fill_(2);

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 0) << "Failed for b > a";
    }
}

// Compare Test Case 4: Edge case - All zeros
TEST_F(CompareKernelTest, AllZeros) {
    a.zero_();
    b.zero_();

    launchCompareKernel(a, b, result);
    auto result_cpu = result.to(torch::kCPU);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(result_cpu[i].item<int>(), 1) << "Failed for all zeros";
    }
}

// Compare Test Case 5: Edge case - Maximum values
TEST_F(CompareKernelTest, MaxValues) {
    uint32_t max_value = (1ULL << 32) - 1;
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

// ADD Test Case 1: Adding two tensors with all elements as 0
TEST_F(LaunchAddKernelTest, AddZeros) {
    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// ADD Test Case 2: Adding two tensors with identical values
TEST_F(LaunchAddKernelTest, AddIdenticalValues) {
    a.fill_(1);
    b.fill_(1);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::full({ n, k }, 2, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// ADD Test Case 3: Adding two tensors with one tensor all zeros
TEST_F(LaunchAddKernelTest, AddWithZeros) {
    b.fill_(42);

    launchAddKernel(a, b, add_result, carry_result);
    torch::Tensor expected_result = torch::full({ n, k }, 42, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::zeros({ n }, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// ADD Test Case 4: Carry propagation across segments
TEST_F(LaunchAddKernelTest, CarryPropagationWithinTensor) {
    a.fill_(UINT32_MAX);
    b.zero_();
    b.select(1, 0).fill_(1);

    launchAddKernel(a, b, add_result, carry_result);

    // Expected result is all zeros, carry is 1
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

    // Expected result is all UINT32_MAX - 1, carry is 1
    torch::Tensor expected_result = torch::full({ n, k }, UINT32_MAX - 1, torch::kUInt32).to(torch::kCUDA);
    torch::Tensor expected_carry = torch::full({ n }, 1, torch::kInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(add_result, expected_result));
    ASSERT_TRUE(torch::allclose(carry_result, expected_carry));
}

// Test suite for subtraction kernel
class LaunchSubKernelTest : public ::testing::Test {
protected:
    torch::Tensor sub_a, sub_b, sub_result;

    void SetUp() override {
        // Allocate tensors
        sub_a = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        sub_b = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        sub_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    }
};

// SUB Test Case 1: Subtracting two tensors with all elements as 0
TEST_F(LaunchSubKernelTest, SubZeros) {
    launchSubKernel(sub_a, sub_b, sub_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(sub_result, expected_result));
}

// SUB Test Case 2: Subtracting two tensors with identical values
TEST_F(LaunchSubKernelTest, SubIdenticalValues) {
    sub_a.fill_(42);
    sub_b.fill_(42);

    launchSubKernel(sub_a, sub_b, sub_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(sub_result, expected_result));
}

// SUB Test Case 3: Subtracting two tensors where the first tensor is greater
TEST_F(LaunchSubKernelTest, SubAGreaterThanB) {
    sub_a.fill_(100);
    sub_b.fill_(42);

    launchSubKernel(sub_a, sub_b, sub_result);
    torch::Tensor expected_result = torch::full({ n, k }, 58, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(sub_result, expected_result));
}


// SUB Test Case 4: Maximum values
TEST_F(LaunchSubKernelTest, SubMaxValues) {
    uint32_t max_value = (1ULL << 32) - 1;
    sub_a.fill_(max_value);
    sub_b.fill_(max_value);

    launchSubKernel(sub_a, sub_b, sub_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(sub_result, expected_result));
}

// SUB Test Case 5: borrow test
TEST_F(LaunchSubKernelTest, MSBAndLSB) {
    uint32_t msb_value = (1ULL << 31); 
    sub_a.fill_(msb_value);

    uint32_t lsb_value = 1;
    sub_b.fill_(lsb_value);

    // Launch the subtraction kernel
    launchSubKernel(sub_a, sub_b, sub_result);

    // Move results to CPU for validation
    auto sub_result_cpu = sub_result.to(torch::kCPU);
    auto sub_a_cpu = sub_a.to(torch::kCPU);
    auto sub_b_cpu = sub_b.to(torch::kCPU);

    // Access the tensors for element-wise comparison
    auto sub_result_accessor = sub_result_cpu.accessor<uint32_t, 2>();
    auto sub_a_accessor = sub_a_cpu.accessor<uint32_t, 2>();
    auto sub_b_accessor = sub_b_cpu.accessor<uint32_t, 2>();

    // Validate results
    uint32_t expected_value = msb_value - lsb_value;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            ASSERT_EQ(sub_result_accessor[i][j], expected_value)
                << "Mismatch at index (" << i << ", " << j << ")";
        }
    }
}

// Test suite for modular addition kernel
class LaunchModAddKernelTest : public ::testing::Test {
protected:
    torch::Tensor a, b, q, modadd_result;

    void SetUp() override {
        // Allocate tensors
        a = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        b = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        q = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
        modadd_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);
    }
};

// MODADD Test Case 1: All inputs are zero
TEST_F(LaunchModAddKernelTest, ModAddZeros) {
    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 2: a + b < q
TEST_F(LaunchModAddKernelTest, SumLessThanQ) {
    a.fill_(42);
    b.fill_(58);
    q.fill_(200);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::full({ n, k }, 100, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 3: a + b == q
TEST_F(LaunchModAddKernelTest, SumEqualsQ) {
    a.fill_(50);
    b.fill_(50);
    q.fill_(100);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 4: a + b > q
TEST_F(LaunchModAddKernelTest, SumGreaterThanQ) {
    a.fill_(150);
    b.fill_(100);
    q.fill_(200);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::full({ n, k }, 50, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 5: a == 0, b == 0, q == max value
TEST_F(LaunchModAddKernelTest, ModAddMaxQWithZeroInputs) {
    q.fill_(UINT32_MAX);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::zeros({ n, k }, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 6: Edge case - Large values near q
TEST_F(LaunchModAddKernelTest, LargeValuesNearQ) {
    a.fill_(UINT32_MAX - 1);
    b.fill_(1);
    q.fill_(UINT32_MAX);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::full({ n, k }, 0, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}

// MODADD Test Case 7: Randomized values with a, b < q
TEST_F(LaunchModAddKernelTest, RandomValuesLessThanQ) {
    a.random_(0, 500);
    b.random_(0, 500);
    q.random_(1000, 2000); // Ensure q > a and q > b

    launchModAddKernel(a, b, q, modadd_result);

    auto a_cpu = a.to(torch::kCPU);
    auto b_cpu = b.to(torch::kCPU);
    auto q_cpu = q.to(torch::kCPU);
    auto modadd_result_cpu = modadd_result.to(torch::kCPU);

    auto a_accessor = a_cpu.accessor<uint32_t, 2>();
    auto b_accessor = b_cpu.accessor<uint32_t, 2>();
    auto q_accessor = q_cpu.accessor<uint32_t, 2>();
    auto result_accessor = modadd_result_cpu.accessor<uint32_t, 2>();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            uint32_t expected_value = (a_accessor[i][j] + b_accessor[i][j]) % q_accessor[i][j];
            ASSERT_EQ(result_accessor[i][j], expected_value)
                << "Mismatch at index (" << i << ", " << j << ")";
        }
    }
}

// MODADD Test Case 8: All inputs are at their maximum values
TEST_F(LaunchModAddKernelTest, MaxValues) {
    a.fill_(UINT32_MAX - 1);
    b.fill_(UINT32_MAX - 1);
    q.fill_(UINT32_MAX);

    launchModAddKernel(a, b, q, modadd_result);
    torch::Tensor expected_result = torch::full({ n, k }, (UINT32_MAX - 2) % UINT32_MAX, torch::kUInt32).to(torch::kCUDA);

    ASSERT_TRUE(torch::allclose(modadd_result, expected_result));
}



