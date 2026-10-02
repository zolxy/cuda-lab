#include "cuda_check.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

__global__ void matrix_mul_naive(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = 1024, K = 1024, N = 1024;
    if (argc == 2) {
        M = K = N = std::atoi(argv[1]);
    } else if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    const std::size_t a_bytes = static_cast<std::size_t>(M) * K * sizeof(float);
    const std::size_t b_bytes = static_cast<std::size_t>(K) * N * sizeof(float);
    const std::size_t c_bytes = static_cast<std::size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), b_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_mul_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    float expected = static_cast<float>(2.0f * K);
    for (std::size_t i = 0; i < h_C.size(); ++i) {
        if (std::abs(h_C[i] - expected) > 1e-3) errors++;
    }

    std::printf("Matrix Mul (%dx%dx%d): errors=%d first=%.1f\n", M, K, N, errors, h_C[0]);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return errors != 0;
}

