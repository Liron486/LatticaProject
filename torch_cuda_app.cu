#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/types.h>
#include <iostream>
#include <random>

#ifdef ENABLE_TESTS
#include <gtest/gtest.h>
#endif

#define CHUNK_BIT_SIZE 32
#define THREADS_PER_BLOCK 256

// Global constexpr for batch size and segments per integer
constexpr int n = 2; // Number of 256-bit integers in the batch
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
    static_assert(BIT_SIZE <= sizeof(scalar_t) * 8, "BIT_SIZE must fit within scalar_t");

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
    static_assert(BIT_SIZE <= 8 * sizeof(scalar_t), "BIT_SIZE must fit within scalar_t");

    uint64_t borrow = 0;
    scalar_t base = (scalar_t)1 << BIT_SIZE;

    for (int i = 0; i < k; ++i) {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i] - (int64_t)borrow;

        if (diff < 0) {
            borrow = 1;
            diff += base; // Adjust diff by adding the base if negative
        }
        else {
            borrow = 0; // Reset borrow if no underflow
        }

        result[i] = (scalar_t)(diff & (base - 1));
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
    add_bigint<scalar_t, BIT_SIZE>(&a[idx][0], &b[idx][0], &add_result[idx][0], carry);
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

    sub_bigint<scalar_t, BIT_SIZE>(&a[idx][0], &b[idx][0], &sub_result[idx][0]);
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
    if (idx >= n * k) return;

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

// Function to allocate and initialize host tensors
void allocateAndInitializeHostTensors(
    torch::Tensor& host_a,
    torch::Tensor& host_b,
    torch::Tensor& host_q)
{
    // Random number generator for initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    uint32_t max_value = (1u << CHUNK_BIT_SIZE) - 1;
    std::uniform_int_distribution<uint32_t> dist(0, max_value);

    // Create host tensors with pinned memory for fast host-to-device transfer
    host_a = torch::empty({ n, k }, torch::TensorOptions().dtype(torch::kUInt32).pinned_memory(true));
    host_b = torch::empty({ n, k }, torch::TensorOptions().dtype(torch::kUInt32).pinned_memory(true));
    host_q = torch::full({ n, k }, max_value, torch::TensorOptions().dtype(torch::kUInt32).pinned_memory(true));

    // Initialize the tensors with random values
    auto* host_a_ptr = host_a.data_ptr<uint32_t>();
    auto* host_b_ptr = host_b.data_ptr<uint32_t>();
    for (int i = 0; i < n * k; ++i) {
        host_a_ptr[i] = dist(gen);
        host_b_ptr[i] = dist(gen);
    }
}


// Function to create streams and events
void createStreamsAndEvents(
    cudaStream_t& stream1, 
    cudaStream_t& stream2, 
    cudaEvent_t& event) 
{
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
    CHECK_CUDA_ERROR(cudaEventCreate(&event));
}

// Function to copy data from host tensors to device tensors
void copyHostToDeviceTensors(
    torch::Tensor& host_a,
    torch::Tensor& host_b,
    torch::Tensor& host_q,
    torch::Tensor& device_a,
    torch::Tensor& device_b,
    torch::Tensor& device_q,
    cudaStream_t stream)
{
    device_a = host_a.to(torch::kCUDA, true, stream);
    device_b = host_b.to(torch::kCUDA, true, stream);
    device_q = host_q.to(torch::kCUDA, true, stream);
}


// Function to launch compare kernel
void launchCompareKernel(
    torch::Tensor& a, 
    torch::Tensor& b, 
    torch::Tensor& compare_result, 
    cudaStream_t stream)
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    compare<uint32_t> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        compare_result.packed_accessor32<int, 1, torch::RestrictPtrTraits>());
}

// Function to launch add kernel
void launchAddKernel(
    torch::Tensor& a, 
    torch::Tensor& b, 
    torch::Tensor& add_result, 
    torch::Tensor& carry_result, 
    cudaStream_t stream)
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
void launchSubKernel(
    torch::Tensor& sub_a, 
    torch::Tensor& sub_b, 
    torch::Tensor& sub_result, 
    cudaStream_t stream) 
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    sub<uint32_t, CHUNK_BIT_SIZE> << <blocks, threads, 0, stream >> > (
        sub_a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        sub_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>());
}

// Function to launch modadd kernel
void launchModAddKernel(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& q,
    torch::Tensor& modadd_result, 
    cudaStream_t stream) 
{
    int threads = THREADS_PER_BLOCK;
    int blocks = (n + threads - 1) / threads;

    modadd<uint32_t, CHUNK_BIT_SIZE> << <blocks, threads, 0, stream >> > (
        a.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        modadd_result.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>());
}

// Function to copy data from device tensors to host tensors
void copyDeviceToHostTensors(
    torch::Tensor& device_add_result,
    torch::Tensor& device_carry_result,
    torch::Tensor& device_sub_result,
    torch::Tensor& device_compare_result,
    torch::Tensor& device_modadd_result,
    torch::Tensor& host_add_result,
    torch::Tensor& host_carry_result,
    torch::Tensor& host_sub_result,
    torch::Tensor& host_compare_result,
    torch::Tensor& host_modadd_result)
{
    host_add_result = device_add_result.to(torch::kCPU, true);
    host_carry_result = device_carry_result.to(torch::kCPU, true);
    host_sub_result = device_sub_result.to(torch::kCPU, true);
    host_compare_result = device_compare_result.to(torch::kCPU, true);
    host_modadd_result = device_modadd_result.to(torch::kCPU, true);
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

void printResults(
    torch::Tensor& host_a,
    torch::Tensor& host_b,
    torch::Tensor& host_compare_result,
    torch::Tensor& host_add_result,
    torch::Tensor& host_carry_result,
    torch::Tensor& host_sub_result,
    torch::Tensor& host_q,
    torch::Tensor& host_modadd_result)
{
    auto* a_ptr = host_a.data_ptr<uint32_t>();
    auto* b_ptr = host_b.data_ptr<uint32_t>();
    auto* q_ptr = host_q.data_ptr<uint32_t>();
    auto* compare_result_ptr = host_compare_result.data_ptr<int>();
    auto* add_result_ptr = host_add_result.data_ptr<uint32_t>();
    auto* carry_result_ptr = host_carry_result.data_ptr<int>();
    auto* sub_result_ptr = host_sub_result.data_ptr<uint32_t>();
    auto* modadd_result_ptr = host_modadd_result.data_ptr<uint32_t>();

    // Print `host_a`
    std::cout << "host_a:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << a_ptr[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print `host_b`
    std::cout << "host_b:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << b_ptr[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print comparison results
    std::cout << "\n\nComparison Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Result[" << i << "] = " << compare_result_ptr[i] << std::endl;
    }

    // Print addition results
    std::cout << "\n\nAddition Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << add_result_ptr[i * k + j] << " ";
        }
        std::cout << "(Carry: " << carry_result_ptr[i] << ")" << std::endl;
    }

    // Print subtraction results
    std::cout << "\n\nSubtraction Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << sub_result_ptr[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print modular addition results
    std::cout << "\n\nModular Addition Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << modadd_result_ptr[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print `host_q`
    std::cout << "\nhost_q:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < k; ++j) {
            std::cout << q_ptr[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char** argv)
{
#ifdef ENABLE_TESTS
    printf("HERE %d\n", ENABLE_TESTS);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#else
    torch::Device device(torch::kCUDA);
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    createStreamsAndEvents(stream1, stream2, event);

    // Host tensors
    torch::Tensor host_a, host_b, host_q;
    torch::Tensor host_add_result, host_carry_result, host_sub_result, host_compare_result, host_modadd_result;

    // Allocate and initialize host tensors
    allocateAndInitializeHostTensors(host_a, host_b, host_q);

    // Device tensors
    torch::Tensor device_a, device_b, device_q;
    torch::Tensor device_add_result, device_carry_result, device_sub_result, device_compare_result, device_modadd_result;

    // Allocate device tensors
    allocateDeviceMemory(device, device_a, device_b, device_compare_result, device_add_result,
        device_carry_result, device_sub_result, device_q, device_modadd_result);

    // Copy data from host to device
    copyHostToDeviceTensors(host_a, host_b, host_q, device_a, device_b, device_q, stream1);

    cudaEventRecord(event, stream1);
    cudaStreamWaitEvent(stream2, event);

    // Launch kernels
    launchCompareKernel(device_a, device_b, device_compare_result, stream1);
    launchAddKernel(device_a, device_b, device_add_result, device_carry_result, stream2);

    torch::Tensor sub_a = torch::zeros({ n, k }, torch::kUInt32).to(device);
    torch::Tensor sub_b = torch::zeros({ n, k }, torch::kUInt32).to(device);

    prepareSubtractionData(device_compare_result, device_a, device_b, sub_a, sub_b, stream1);
    launchSubKernel(sub_a, sub_b, device_sub_result, stream1);
    launchModAddKernel(device_a, device_b, device_q, device_modadd_result, stream2);

    // Copy results from device to host
    copyDeviceToHostTensors(device_add_result, device_carry_result, device_sub_result, device_compare_result,
        device_modadd_result, host_add_result, host_carry_result, host_sub_result,
        host_compare_result, host_modadd_result);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print results
    printResults(host_a, host_b, host_compare_result, host_add_result, host_carry_result, host_sub_result, host_q, host_modadd_result);

    // Free resources
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);

    return 0;
#endif
}
