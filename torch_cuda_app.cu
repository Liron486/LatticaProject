#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/types.h>
#include <iostream>
#include <random>

// Kernel to compare two 256-bit integers
template <typename scalar_t>
__global__ void compare(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    bool is_greater_or_equal = true;
    for (int i = 7; i >= 0; --i) {
        if (a[idx][i] > b[idx][i]) {
            break;
        }
        else if (a[idx][i] < b[idx][i]) {
            is_greater_or_equal = false;
            break;
        }
    }

    result[idx] = is_greater_or_equal ? 1 : 0;
}

// Kernel to add two 256-bit integers
// Generalized add kernel with configurable segment size
template <typename scalar_t, int BIT_SIZE>
__global__ void add(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> add_result,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> carry_out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    // Calculate the modulo and carry limits based on the bit size
    const uint64_t MODULO = (uint64_t)1 << BIT_SIZE; // 2^BIT_SIZE
    const uint64_t CARRY_MASK = MODULO - 1;          // Mask for the carry bits

    scalar_t carry = 0; // Initialize carry
    for (int i = 0; i < 8; ++i) { // Process each segment
        uint64_t sum = (uint64_t)a[idx][i] + (uint64_t)b[idx][i] + carry;
        add_result[idx][i] = (scalar_t)(sum & CARRY_MASK); // Apply modulo operation
        carry = sum >> BIT_SIZE; // Extract carry for the next segment
    }

    // Save the final carry
    carry_out[idx] = (int)carry;
}


int main() {
    int n = 1; // Number of 256-bit integers in the batch
    int k = 8; // Number of 32-bit segments per integer

    torch::Device device(torch::kCUDA);

    // Allocate pinned host memory for overlapping
    uint32_t* host_a = nullptr;
    uint32_t* host_b = nullptr;
    int* host_compare_result = nullptr;
    uint32_t* host_add_result = nullptr;
    int* host_carry_result = nullptr;

    cudaMallocHost(&host_a, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_b, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_compare_result, n * sizeof(int));
    cudaMallocHost(&host_add_result, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_carry_result, n * sizeof(int));

    // Initialize input on host
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_int_distribution<uint32_t> dist(0, 0xF); // Full 32-bit range

    for (int i = 0; i < n * k; ++i) {
        host_a[i] = dist(gen); // Generate random 32-bit non-negative integer
        host_b[i] = dist(gen);
    }

    // Allocate device memory
    torch::Tensor a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor compare_result = torch::zeros({ n }, torch::kInt32).to(device);
    torch::Tensor add_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor carry_result = torch::zeros({ n }, torch::kInt32).to(device);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create a CUDA event for synchronization
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Asynchronous data transfer in stream1
    cudaMemcpyAsync(a.data_ptr<uint32_t>(), host_a, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(b.data_ptr<uint32_t>(), host_b, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream1);

    // Record an event in stream1 after data transfer
    cudaEventRecord(event, stream1);

    // Make stream2 wait for the data transfer to complete
    cudaStreamWaitEvent(stream2, event, 0);

    // Launch kernels
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Compare kernel in stream1
    compare<uint32_t> << <blocks, threads, 0, stream1 >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        n
        );

    // Add kernel in stream2
    add<uint32_t, 4> << <blocks, threads, 0, stream2 >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        add_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        carry_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        n
        );

    // Asynchronous data transfer back to host
    cudaMemcpyAsync(host_compare_result, compare_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(host_add_result, add_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(host_carry_result, carry_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print `host_a` and `host_b`
    std::cout << "host_a:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_a[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n\nhost_b:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_b[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print comparison results
    std::cout << "\n\nComparison Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Result[" << i << "] = " << host_compare_result[i] << std::endl;
    }

    // Print addition results
    std::cout << "\n\nAddition Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_add_result[i * k + j] << " ";
        }
        std::cout << "(Carry: " << host_carry_result[i] << ")" << std::endl;
    }

    // Free resources
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_compare_result);
    cudaFreeHost(host_add_result);
    cudaFreeHost(host_carry_result);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);

    return 0;
}
