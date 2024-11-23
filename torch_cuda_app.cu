#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/types.h>
#include <iostream>
#include <random>

#define CHUNK_BIT_SIZE 32
#define THREADS_PER_BLOCK 256

// Global constexpr for batch size and segments per integer
constexpr int n = 8; // Number of 256-bit integers in the batch
constexpr int k = 8; // Number of 32-bit segments per integer

// Error-checking macro
#define CHECK_CUDA_ERROR(call)                                                                 \
    do {                                                                                       \
        cudaError_t error = call;                                                              \
        if (error != cudaSuccess) {                                                            \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__                       \
                      << " (" << cudaGetErrorString(error) << ")" << std::endl;                \
            return;                                                                            \
        }                                                                                      \
    } while (0)

template <typename scalar_t>
__device__ int compare_bigint(
    const scalar_t* a,
    const scalar_t* b)
{
    for (int i = k - 1; i >= 0; --i) {
        if (a[i] > b[i]) {
            return 1;
        }
        else if (a[i] < b[i]) {
            return -1;
        }
    }
    return 0; // a == b
}

template <typename scalar_t, int BIT_SIZE>
__device__ void add_bigint(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* result,
    scalar_t& carry_result)
{
    const uint64_t modulo = (1ULL << BIT_SIZE);
    const uint64_t carry_mask = modulo - 1;

    uint64_t carry = 0;
    for (int i = 0; i < k; ++i) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        result[i] = (scalar_t)(sum & carry_mask);
        carry = (sum >> BIT_SIZE);
    }
    carry_result = static_cast<scalar_t>(carry & carry_mask);
}

template <typename scalar_t, int BIT_SIZE>
__device__ void sub_bigint(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* result)
{
    uint64_t borrow = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        if (a[i] < b[i] + borrow) {
            borrow = 1;
            diff += (uint64_t)1 << BIT_SIZE;
        }
        else {
            borrow = 0;
        }
        result[i] = (scalar_t)(diff & ((1ULL << BIT_SIZE) - 1));
    }
}

// Kernel to compare two 256-bit integers
template <typename scalar_t>
__global__ void compare(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = compare_bigint<scalar_t>(&a[idx][0], &b[idx][0]) >= 0 ? 1 : 0;
}

// Kernel to add two 256-bit integers
template <typename scalar_t, int BIT_SIZE>
__global__ void add(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> add_result,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> carry_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    scalar_t carry = 0;
    scalar_t sum[k] = { 0 };

    add_bigint<scalar_t, BIT_SIZE>(&a[idx][0], &b[idx][0], sum, carry);

    for (int i = 0; i < k; ++i) {
        add_result[idx][i] = sum[i];
    }
    carry_out[idx] = static_cast<int>(carry & 1);
}


// Kernel to subtract two 256-bit integers
template <typename scalar_t, int BIT_SIZE>
__global__ void sub(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sub_result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    scalar_t diff[k];
    sub_bigint<scalar_t, BIT_SIZE>(&a[idx][0], &b[idx][0], diff);

    for (int i = 0; i < k; ++i) {
        sub_result[idx][i] = diff[i];
    }
}

// Kernel to perform modular addition
template <typename scalar_t, int BIT_SIZE>
__global__ void modadd(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> q,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    scalar_t carry;
    scalar_t sum[k];
    scalar_t sub_res[k];

    // Perform addition
    add_bigint<scalar_t, BIT_SIZE>(&a[idx][0], &b[idx][0], sum, carry);

    int cmp = compare_bigint<scalar_t>(&sum[0], &q[idx][0]);

    if (carry > 0 || cmp >= 0) {
        // sum >= q or there was an overflow, subtract q
        sub_bigint<scalar_t, BIT_SIZE>(&sum[0], &q[idx][0], sub_res);
        for (int i = 0; i < k; ++i) {
            result[idx][i] = sub_res[i];
        }
    }
    else {
        // sum < q
        for (int i = 0; i < k; ++i) {
            result[idx][i] = sum[i];
        }
    }
}

__global__ void prepare_subtraction_data_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> compare_result,
    const torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> sub_a,
    torch::PackedTensorAccessor32<uint32_t, 2, torch::RestrictPtrTraits> sub_b)
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
void allocateAndInitializeHostMemory(
    uint32_t*& host_a,
    uint32_t*& host_b,
    int*& host_compare_result,
    uint32_t*& host_add_result,
    int*& host_carry_result,
    uint32_t*& host_sub_result,
    uint32_t*& host_q,
    uint32_t*& host_modadd_result)
{
    CHECK_CUDA_ERROR(cudaMallocHost(&host_a, n * k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_b, n * k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_compare_result, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_add_result, n * k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_carry_result, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_sub_result, n * k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_q, n * k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMallocHost(&host_modadd_result, n * k * sizeof(uint32_t)));

    std::random_device rd;
    std::mt19937 gen(rd());
    uint32_t max_value = (1u << CHUNK_BIT_SIZE) - 1;
    std::uniform_int_distribution<uint32_t> dist(0, max_value);

    for (int i = 0; i < n * k; ++i) {
        host_a[i] = dist(gen);
        host_b[i] = dist(gen);
        host_q[i] = max_value;
    }
}

// Function to allocate device memory
void allocateDeviceMemory(
    torch::Device device,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& compare_result,
    torch::Tensor& add_result,
    torch::Tensor& carry_result,
    torch::Tensor& sub_result,
    torch::Tensor& q,
    torch::Tensor& modadd_result)
{
    a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    compare_result = torch::zeros({ n }, torch::kInt32).to(device);
    add_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
    carry_result = torch::zeros({ n }, torch::kInt32).to(device);
    sub_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
    q = torch::zeros({ n, k }, torch::kUInt32).to(device); 
    modadd_result = torch::zeros({ n, k }, torch::kUInt32).to(device);
}

// Function to create streams and events
void createStreamsAndEvents(cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& event) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
    CHECK_CUDA_ERROR(cudaEventCreate(&event));
}

// Function to copy data from host to device asynchronously
void copyHostToDeviceAsync(
    uint32_t* host_a,
    uint32_t* host_b,
    uint32_t* host_q,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& q, 
    cudaStream_t stream)
{
    // Asynchronous data transfer in stream
    cudaMemcpyAsync(a.data_ptr<uint32_t>(), host_a, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b.data_ptr<uint32_t>(), host_b, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(q.data_ptr<uint32_t>(), host_q, n * k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
}

// Function to launch compare kernel
void launchCompareKernel(torch::Tensor& a, torch::Tensor& b, torch::Tensor& compare_result, cudaStream_t stream)
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    compare<uint32_t> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>());
}

// Function to launch add kernel
void launchAddKernel(torch::Tensor& a, torch::Tensor& b, torch::Tensor& add_result, torch::Tensor& carry_result, cudaStream_t stream)
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    add<uint32_t, CHUNK_BIT_SIZE> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        add_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        carry_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>());
}

// Function to prepare data for subtraction on the device
void prepareSubtractionData(
    torch::Tensor& compare_result,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& sub_a,
    torch::Tensor& sub_b,
    cudaStream_t stream)
{
    int threads = THREADS_PER_BLOCK;
    int blocks = ((n * k) + threads - 1) / threads;

    prepare_subtraction_data_kernel << <blocks, threads, 0, stream >> > (
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>());
}

// Function to launch subtraction kernel
void launchSubKernel(torch::Tensor& sub_a, torch::Tensor& sub_b, torch::Tensor& sub_result, cudaStream_t stream) 
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    sub<uint32_t, CHUNK_BIT_SIZE> << <blocks, threads, 0, stream >> > (
        sub_a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>());
}

// Function to launch modadd kernel
void launchModAddKernel(torch::Tensor& a, torch::Tensor& b, torch::Tensor& q, torch::Tensor& modadd_result, cudaStream_t stream) 
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    modadd<uint32_t, CHUNK_BIT_SIZE> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        modadd_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>());
}

// Updated function to copy data from device to host asynchronously
void copyDeviceToHostAsync(
    uint32_t* host_add_result,
    int* host_carry_result,
    uint32_t* host_sub_result,
    int* host_compare_result,
    uint32_t* host_modadd_result,
    torch::Tensor& add_result,
    torch::Tensor& carry_result,
    torch::Tensor& sub_result,
    torch::Tensor& compare_result,
    torch::Tensor& modadd_result, 
    cudaStream_t stream_add,
    cudaStream_t stream_sub)
{
    // Copy addition results
    cudaMemcpyAsync(host_add_result, add_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_add);
    cudaMemcpyAsync(host_carry_result, carry_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream_add);

    // Copy subtraction results
    cudaMemcpyAsync(host_sub_result, sub_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_sub);

    // Copy comparison results
    cudaMemcpyAsync(host_compare_result, compare_result.data_ptr<int>(), n * sizeof(int), cudaMemcpyDeviceToHost, stream_sub);

    // Copy modadd results
    cudaMemcpyAsync(host_modadd_result, modadd_result.data_ptr<uint32_t>(), n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_add);
}

// Function to print results
void printResults(
    uint32_t* host_a,
    uint32_t* host_b,
    int* host_compare_result,
    uint32_t* host_add_result,
    int* host_carry_result,
    uint32_t* host_sub_result,
    uint32_t* host_q,
    uint32_t* host_modadd_result)
{
    // Print `host_a` and `host_b`
    std::cout << "host_a:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_a[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "host_b:" << std::endl;
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
        std::cout << std::dec << std::endl;
    }

    // Print modadd results
    std::cout << "\n\nModular Addition Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_modadd_result[i * k + j] << " ";
        }
        std::cout << std::dec << std::endl;
    }

    // Print 'host_q'
    std::cout << "\nhost_q:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << host_q[i * k + j] << " ";
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
    uint32_t* host_q,
    uint32_t* host_modadd_result,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaEvent_t event)
{
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_compare_result);
    cudaFreeHost(host_add_result);
    cudaFreeHost(host_carry_result);
    cudaFreeHost(host_sub_result);
    cudaFreeHost(host_q);
    cudaFreeHost(host_modadd_result);

    if (stream1) cudaStreamDestroy(stream1);
    if (stream2) cudaStreamDestroy(stream2);
    if (event) cudaEventDestroy(event);
}

int main() 
{
    torch::Device device(torch::kCUDA);

    // Host memory pointers
    uint32_t* host_a = nullptr;
    uint32_t* host_b = nullptr;
    int* host_compare_result = nullptr;
    uint32_t* host_add_result = nullptr;
    int* host_carry_result = nullptr;
    uint32_t* host_sub_result = nullptr;
    uint32_t* host_q = nullptr;
    uint32_t* host_modadd_result = nullptr;

    // Allocate and initialize host memory
    allocateAndInitializeHostMemory(host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result, host_q, host_modadd_result);

    // Device tensors
    torch::Tensor a, b, compare_result, add_result, carry_result, sub_result, q, modadd_result;

    // Allocate device memory
    allocateDeviceMemory(device, a, b, compare_result, add_result, carry_result, sub_result, q, modadd_result);

    // Create streams and events
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    createStreamsAndEvents(stream1, stream2, event);

    // Copy data from host to device asynchronously
    copyHostToDeviceAsync(host_a, host_b, host_q, a, b, q, stream1);

    // Record an event in stream1 after data transfer
    cudaEventRecord(event, stream1);

    // Make stream2 wait for the data transfer to complete
    cudaStreamWaitEvent(stream2, event, 0);

    launchCompareKernel(a, b, compare_result, stream1);
    launchAddKernel(a, b, add_result, carry_result, stream2);

    //// Prepare data for subtraction on the device
    torch::Tensor sub_a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor sub_b = torch::zeros({ n, k }, torch::kUInt32).to(device);
    prepareSubtractionData(compare_result, a, b, sub_a, sub_b, stream1);

    launchSubKernel(sub_a, sub_b, sub_result, stream1);
    launchModAddKernel(a, b, q, modadd_result, stream2);

    // Copy results from device to host asynchronously
    copyDeviceToHostAsync(host_add_result, host_carry_result, host_sub_result, host_compare_result, host_modadd_result,
        add_result, carry_result, sub_result, compare_result, modadd_result, stream2, stream1);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print results
    printResults(host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result, host_q, host_modadd_result);

    // Free resources
    freeResources(host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result, host_q, host_modadd_result, stream1, stream2, event);

    return 0;
}
