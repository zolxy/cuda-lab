#include "cuda_check.cuh"
#include <iostream>
#include <vector>
#include <cstdio>

#define RADIUS 1
#define BLOCK_SIZE 16

__global__ void stencil_2d(const float *in, float *out, int width, int height) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_x = threadIdx.x + RADIUS;
    int local_y = threadIdx.y + RADIUS;

    // Load main element
    if (x < width && y < height) {
        smem[local_y][local_x] = in[y * width + x];
    } else {
        smem[local_y][local_x] = 0.0f;
    }

    // Load halo regions (top, bottom, left, right)
    if (threadIdx.y < RADIUS) {
        smem[local_y - RADIUS][local_x] = (y >= RADIUS && x < width) ? in[(y - RADIUS) * width + x] : 0.0f;
        smem[local_y + BLOCK_SIZE][local_x] = (y + BLOCK_SIZE < height && x < width) ? in[(y + BLOCK_SIZE) * width + x] : 0.0f;
    }
    if (threadIdx.x < RADIUS) {
        smem[local_y][local_x - RADIUS] = (x >= RADIUS && y < height) ? in[y * width + (x - RADIUS)] : 0.0f;
        smem[local_y][local_x + BLOCK_SIZE] = (x + BLOCK_SIZE < width && y < height) ? in[y * width + (x + BLOCK_SIZE)] : 0.0f;
    }

    __syncthreads();

    // Apply 5-point stencil (center + 4 neighbors)
    if (x < width && y < height) {
        float sum = smem[local_y][local_x] * 4.0f
                  - smem[local_y - 1][local_x]
                  - smem[local_y + 1][local_x]
                  - smem[local_y][local_x - 1]
                  - smem[local_y][local_x + 1];
        out[y * width + x] = sum;
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);

    std::vector<float> h_in(width * height, 1.0f);
    std::vector<float> h_out(width * height, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    stencil_2d<<<blocks, threads>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::printf("2D Stencil (512x512): Output center element = %.2f\n", h_out[(height / 2) * width + (width / 2)]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
