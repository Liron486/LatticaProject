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

// Kernel to subtract two 256-bit integers
template <typename scalar_t, int BIT_SIZE>
__global__ void sub(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sub_result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    scalar_t borrow = 0; // Initialize borrow
    for (int i = 0; i < 8; ++i) { // Process each segment
        uint64_t diff = (uint64_t)a[idx][i] - (uint64_t)b[idx][i] - borrow;
        if (a[idx][i] < b[idx][i] + borrow) {
            borrow = 1; // Borrow from the next higher segment
            diff += (uint64_t)1 << BIT_SIZE; // Add back the base
        }
        else {
            borrow = 0; // No borrow needed
        }
        sub_result[idx][i] = (scalar_t)(diff & ((1 << BIT_SIZE) - 1)); // Store result
    }
}

__global__ void prepare_subtraction_data_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> compare_result,
    const torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> sub_a,
    torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> sub_b,
    int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) {
        return;
    }

    int row = idx / k;
    int col = idx % k;

    if (compare_result[row] == 1) {
        sub_a[row][col] = a[row][col];
        sub_b[row][col] = b[row][col];
    }
    else {
        sub_a[row][col] = b[row][col];
        sub_b[row][col] = a[row][col];
    }
}

// Function to allocate and initialize host memory
void allocateAndInitializeHostMemory(int n, int k,
    uint32_t*& host_a,
    uint32_t*& host_b,
    int*& host_compare_result,
    uint32_t*& host_add_result,
    int*& host_carry_result,
    uint32_t*& host_sub_result) {
    // Allocate pinned host memory for overlapping
    cudaMallocHost(&host_a, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_b, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_compare_result, n * sizeof(int));
    cudaMallocHost(&host_add_result, n * k * sizeof(uint32_t));
    cudaMallocHost(&host_carry_result, n * sizeof(int));
    cudaMallocHost(&host_sub_result, n * k * sizeof(uint32_t));

    // Initialize input on host
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_int_distribution<uint32_t> dist(0, 0xF); // Full 32-bit range

    for (int i = 0; i < n * k; ++i) {
        host_a[i] = dist(gen); // Generate random 32-bit non-negative integer
        host_b[i] = dist(gen);
    }
}

// Function to allocate device memory
void allocateDeviceMemory(int n, int k, torch::Device device,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& compare_result,
    torch::Tensor& add_result,
    torch::Tensor& carry_result,
    torch::Tensor& sub_result) {
    // Allocate device memory
    a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    compare_result = torch::zeros({ n }, torch::kInt32).to(device);
    add_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
    carry_result = torch::zeros({ n }, torch::kInt32).to(device);
    sub_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
}

// Function to create streams and events
void createStreamsAndEvents(cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& event) {
    // Create CUDA streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create a CUDA event for synchronization
    cudaEventCreate(&event);
}

// Function to copy data from host to device asynchronously
void copyHostToDeviceAsync(int n, int k,
    uint32_t* host_a,
    uint32_t* host_b,
    torch::Tensor& a,
    torch::Tensor& b,
    cudaStream_t stream) {
    // Asynchronous data transfer in stream
    cudaMemcpyAsync(a.data_ptr<uint32_t>(), host_a, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b.data_ptr<uint32_t>(), host_b, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
}

// Function to launch compare kernel
void launchCompareKernel(int n, torch::Tensor& a, torch::Tensor& b, torch::Tensor& compare_result, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    compare<uint32_t> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        n
        );
}

// Function to launch add kernel
void launchAddKernel(int n, torch::Tensor& a, torch::Tensor& b, torch::Tensor& add_result, torch::Tensor& carry_result, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add<uint32_t, 4> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        add_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        carry_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        n
        );
}

// Function to prepare data for subtraction on the device
void prepareSubtractionData(
    int n, int k,
    torch::Tensor& compare_result,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& sub_a,
    torch::Tensor& sub_b,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = ((n * k) + threads - 1) / threads;

    prepare_subtraction_data_kernel<<<blocks, threads, 0, stream>>>(
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        n, k);
}

// Function to launch subtraction kernel
void launchSubKernel(int n, torch::Tensor& sub_a, torch::Tensor& sub_b, torch::Tensor& sub_result, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sub<uint32_t, 4> << <blocks, threads, 0, stream >> > (
        sub_a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        n
        );
}

// Updated function to copy data from device to host asynchronously
void copyDeviceToHostAsync(
    int n, int k,
    uint32_t* host_add_result,
    int* host_carry_result,
    uint32_t* host_sub_result,
    int* host_compare_result, // Added this parameter
    torch::Tensor& add_result,
    torch::Tensor& carry_result,
    torch::Tensor& sub_result,
    torch::Tensor& compare_result, // Added this parameter
    cudaStream_t stream_add,
    cudaStream_t stream_sub
)
{
    // Copy addition results
    cudaMemcpyAsync(host_add_result, add_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_add);
    cudaMemcpyAsync(host_carry_result, carry_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream_add);

    // Copy subtraction results
    cudaMemcpyAsync(host_sub_result, sub_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_sub);

    // Copy comparison results
    cudaMemcpyAsync(host_compare_result, compare_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream_sub);
}

// Function to print results
void printResults(
    int n, int k,
    uint32_t* host_a,
    uint32_t* host_b,
    int* host_compare_result,
    uint32_t* host_add_result,
    int* host_carry_result,
    uint32_t* host_sub_result
) {
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

    // Print subtraction results
    std::cout << "\n\nSubtraction Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_sub_result[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to free resources
void freeResources(
    uint32_t* host_a,
    uint32_t* host_b,
    int* host_compare_result,
    uint32_t* host_add_result,
    int* host_carry_result,
    uint32_t* host_sub_result,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaEvent_t event
) {
    // Free resources
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_compare_result);
    cudaFreeHost(host_add_result);
    cudaFreeHost(host_carry_result);
    cudaFreeHost(host_sub_result);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);
}

int main() {
    int n = 1; // Number of 256-bit integers in the batch
    int k = 8; // Number of 32-bit segments per integer
    
    torch::Device device(torch::kCUDA);

    // Host memory pointers
    uint32_t* host_a = nullptr;
    uint32_t* host_b = nullptr;
    int* host_compare_result = nullptr;
    uint32_t* host_add_result = nullptr;
    int* host_carry_result = nullptr;
    uint32_t* host_sub_result = nullptr;

    // Allocate and initialize host memory
    allocateAndInitializeHostMemory(n, k, host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result);

    // Device tensors
    torch::Tensor a, b, compare_result, add_result, carry_result, sub_result;

    // Allocate device memory
    allocateDeviceMemory(n, k, device, a, b, compare_result, add_result, carry_result, sub_result);

    // Create streams and events
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    createStreamsAndEvents(stream1, stream2, event);

    // Copy data from host to device asynchronously
    copyHostToDeviceAsync(n, k, host_a, host_b, a, b, stream1);

    // Record an event in stream1 after data transfer
    cudaEventRecord(event, stream1);

    // Make stream2 wait for the data transfer to complete
    cudaStreamWaitEvent(stream2, event, 0);

    // Launch kernels
    launchCompareKernel(n, a, b, compare_result, stream1);
    launchAddKernel(n, a, b, add_result, carry_result, stream2);

    // Prepare data for subtraction on the device
    torch::Tensor sub_a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor sub_b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    prepareSubtractionData(n, k, compare_result, a, b, sub_a, sub_b, stream1);

    // Launch subtraction kernel
    launchSubKernel(n, sub_a, sub_b, sub_result, stream1);

    // Copy results from device to host asynchronously
    copyDeviceToHostAsync(n, k, host_add_result, host_carry_result, host_sub_result, host_compare_result,
        add_result, carry_result, sub_result, compare_result, stream2, stream1);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print results
    printResults(n, k, host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result);

    // Free resources
    freeResources(host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result, stream1, stream2, event);

    return 0;
}
