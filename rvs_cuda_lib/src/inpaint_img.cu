#include "inpaint_img.cuh"

__device__ bool operator==(float3 a, float3 b) {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__global__ void initializeMapKernel(float3* devColor, int3* map, int imgHeight, int imgWidth, float3 empty_color)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    map[threadId] = make_int3(tx, ty, devColor[threadId] == empty_color ? imgWidth + imgHeight : 0);
}

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
