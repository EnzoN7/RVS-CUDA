#include "inpaint_img.cuh"

/**
* Auteur : Enzo DI MARIA
*/

/**
 * @brief Overloads the equality operator to compare two float3 structures.
 *
 * @param a First float3 structure.
 * @param b Second float3 structure.
 * @return true if both float3 structures are equal in all components, otherwise false.
 */
__device__ bool operator==(float3 a, float3 b)
{
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

/**
 * @brief CUDA kernel that initializes a map for inpainting by marking empty pixels.
 *
 * @param devColor Pointer to the device memory containing the image colors.
 * @param map Pointer to the device memory where the map will be stored.
 * @param imgHeight Height of the image.
 * @param imgWidth Width of the image.
 * @param empty_color The color used to identify empty pixels (to be inpainted).
 */
__global__ void initializeMapKernel(float3* devColor, int3* map, int imgHeight, int imgWidth, float3 empty_color)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    map[threadId] = make_int3(tx, ty, devColor[threadId] == empty_color ? imgWidth + imgHeight : 0);
}

/**
 * @brief CUDA kernel that computes the nearest non-empty pixel for each empty pixel in the map.
 *
 * @param map Pointer to the device memory containing the map.
 * @param imgHeight Height of the image.
 * @param imgWidth Width of the image.
 * @param change Pointer to a boolean flag that indicates if a change was made (used for iterative processing).
 */
__global__ void computeNearestKernel(int3* map, int imgHeight, int imgWidth, bool* change)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    int3 pos = map[threadId];

    if (pos.z > 0)
    {
        for (int dx = fmaxf(tx - 1, 0); dx < fminf(tx + 2, imgWidth); ++dx)
        {
            for (int dy = fmaxf(ty - 1, 0); dy < fminf(ty + 2, imgHeight); ++dy)
            {
                if (abs(tx - dx) + abs(ty - dy) == 1)
                {
                    int neighborIdx = dy * imgWidth + dx;
                    int3 path = map[neighborIdx];
                    if (path.z + 1 < map[threadId].z)
                    {
                        map[threadId] = make_int3(path.x, path.y, path.z + 1);
                        *change = true;
                    }
                }
            }
        }
    }
}

/**
 * @brief CUDA kernel that performs inpainting by filling empty pixels with the color of their nearest non-empty neighbor.
 *
 * @param devColor Pointer to the device memory containing the image colors.
 * @param map Pointer to the device memory containing the map with nearest non-empty pixel information.
 * @param imgHeight Height of the image.
 * @param imgWidth Width of the image.
 * @param empty_color The color used to identify empty pixels (to be inpainted).
 */
__global__ void inpaintKernel(float3* devColor, int3* map, int imgHeight, int imgWidth, float3 empty_color)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    if (devColor[threadId] == empty_color)
    {
        int3 pix = map[threadId];
        int srcIdx = pix.y * imgWidth + pix.x;
        devColor[threadId] = devColor[srcIdx];
    }
}

/**
 * @brief Performs image inpainting on a color image using CUDA. The inpainting fills in empty (green) pixels based on the nearest non-empty pixels.
 *
 * @param devColor Pointer to the device memory containing the color image.
 * @param dstSize The size of the image.
 *
 * @throws std::runtime_error If a CUDA error occurs.
 */
void inpaintImg(float3*& devColor, cv::Size dstSize)
{
    int imgHeight = dstSize.height;
    int imgWidth = dstSize.width;
    int imgSize = imgHeight * imgWidth;

    int3* devMap;
    bool change = true;
    bool* devChange;

    float3 empty_color = make_float3(0.0f, 1.0f, 0.0f);

    cudaMalloc(&devMap, imgSize * sizeof(int3));
    cudaMalloc(&devChange, sizeof(bool));

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    initializeMapKernel <<<gridDim, blockDim>>> (devColor, devMap, imgHeight, imgWidth, empty_color);
    cudaDeviceSynchronize();

    while (change)
    {
        change = false;
        cudaMemcpy(devChange, &change, sizeof(bool), cudaMemcpyHostToDevice);
        computeNearestKernel <<<gridDim, blockDim>>> (devMap, imgHeight, imgWidth, devChange);
        cudaDeviceSynchronize();
        cudaMemcpy(&change, devChange, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    inpaintKernel <<<gridDim, blockDim>>> (devColor, devMap, imgHeight, imgWidth, empty_color);
    cudaDeviceSynchronize();

    cudaFree(devMap);
    cudaFree(devChange);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}
