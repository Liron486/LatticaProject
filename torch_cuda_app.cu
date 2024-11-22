#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/types.h>
#include <iostream>
#include <random>

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

    bool is_greater_or_equal = false;
    for (int i = 7; i >= 0; --i) {
        if (a[idx][i] > b[idx][i]) {
            is_greater_or_equal = true;
            break;
        }
        else if (a[idx][i] < b[idx][i]) {
            is_greater_or_equal = false;
            break;
        }
    }

    result[idx] = is_greater_or_equal ? 1 : 0;
}

int main() {
    int n = 4; // Number of 256-bit integers in the batch
    int k = 8; // Number of 32-bit segments per integer

    torch::Device device(torch::kCUDA);

    // Allocate pinned host memory for overlapping
    uint32_t* host_a = nullptr;
    uint32_t* host_b = nullptr;
    int* host_compare_result = nullptr;

    cudaMallocHost(&host_a, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_b, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_compare_result, n * sizeof(int));

    // Initialize input on host
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF); // Full 32-bit range

    for (int i = 0; i < n * k; ++i) {
        host_a[i] = dist(gen); // Generate random 32-bit non-negative integer
        host_b[i] = dist(gen);
    }

    // Allocate device memory (using torch::kUInt32 to match uint32_t)
    torch::Tensor a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor compare_result = torch::zeros({ n }, torch::kInt32).to(device);

    // Create CUDA streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronous data transfer
    cudaMemcpyAsync(a.data_ptr<uint32_t>(), host_a, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b.data_ptr<uint32_t>(), host_b, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Launch kernel (on stream for synchronization simplicity)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compare<uint32_t> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        n
        );

    // Asynchronous data transfer back to host
    cudaMemcpyAsync(host_compare_result, compare_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Synchronize streams
    cudaStreamSynchronize(stream);

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

    // Print results
    std::cout << "\n\nComparison Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Result[" << i << "] = " << host_compare_result[i] << std::endl;
    }

    // Free resources
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_compare_result);
    cudaStreamDestroy(stream);

    return 0;
}
