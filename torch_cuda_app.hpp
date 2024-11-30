#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/cuda.h>
#include <iostream>

// Constants
constexpr int n = 2; // Number of 256-bit integers in the batch
constexpr int k = 8; // Number of 32-bit segments per integer

// Function to launch compare kernel
void launchCompareKernel(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& compare_result,
    cudaStream_t stream = 0);

// Function to launch add kernel
void launchAddKernel(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& add_result,
    torch::Tensor& carry_result,
    cudaStream_t stream = 0);

// Function to launch subtraction kernel
void launchSubKernel(
    torch::Tensor& sub_a,
    torch::Tensor& sub_b,
    torch::Tensor& sub_result,
    cudaStream_t stream = 0);

// Function to launch modular addition kernel
void launchModAddKernel(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& q,
    torch::Tensor& modadd_result,
    cudaStream_t stream = 0);





