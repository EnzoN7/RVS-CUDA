#include "scale_uv.cuh"

/**
* Auteur : Enzo DI MARIA
*/

/**
 * @brief CUDA kernel that scales UV coordinates by a given factor.
 *
 * @param virtualUV Pointer to the array of UV coordinates (device memory).
 * @param scale Scaling factor to apply to each UV coordinate.
 * @param inputSize The total number of UV coordinates to process.
 */
__global__ void scaleUVKernel(float2* virtualUV, float scale, int inputSize)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < inputSize)
    {
        virtualUV[threadId].x *= scale;
        virtualUV[threadId].y *= scale;
    }
}

/**
 * @brief Scales UV coordinates by a given factor using a CUDA kernel.
 *
 * @param devVirtualUV Pointer to the device memory array of UV coordinates.
 * @param scale Scaling factor to apply to each UV coordinate.
 * @param inputSize The total number of UV coordinates to process.
 *
 * @throws std::runtime_error If a CUDA error occurs.
 */
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
