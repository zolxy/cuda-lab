#include "cuda_check.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void reduce_sum(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {
    int n = 1000003;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t block_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = (i % 7) * 0.125f;
    }

    double cpu_sum = 0.0;
    for (int i = 0; i < n; ++i) cpu_sum += h_in[i];

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), block_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    std::size_t shared_bytes = threads * sizeof(float);
    reduce_sum<<<blocks, threads, shared_bytes>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_out(blocks);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, block_bytes, cudaMemcpyDeviceToHost));

    double gpu_sum = 0.0;
    for (int i = 0; i < blocks; ++i) gpu_sum += h_out[i];

    bool ok = (std::abs(cpu_sum - gpu_sum) < 1e-3);
    std::printf("Reduction: n=%d blocks=%d threads=%d ok=%s\n", n, blocks, threads, ok ? "true" : "false");
    std::printf("CPU sum=%.5f GPU sum=%.5f\n", cpu_sum, gpu_sum);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
