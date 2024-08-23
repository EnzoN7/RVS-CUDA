#include "scale_uv.cuh"

__global__ void scaleUVKernel(float2* virtualUV, float scale, int inputSize)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < inputSize)
    {
        virtualUV[threadId].x *= scale;
        virtualUV[threadId].y *= scale;
    }
}

void scaleUV(float2*& devVirtualUV, float scale, int inputSize)
{
    int threads_per_block = 128;

    int blocks_per_grid = (inputSize + threads_per_block - 1) / threads_per_block;

    scaleUVKernel <<<blocks_per_grid, threads_per_block>>> (devVirtualUV, scale, inputSize);

    cudaDeviceSynchronize();

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}
