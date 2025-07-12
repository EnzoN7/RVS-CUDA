/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
    Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "inpaint_img.cuh"


template<typename color_t>
__device__ color_t getGreen();

template<>
__device__ float3 getGreen<float3>()
{
    return make_float3(0.0f, 1.0f, 0.0f);
}

template<>
__device__ double3 getGreen<double3>()
{
    return make_double3(0.0, 1.0, 0.0);
}

template<>
__device__ half3 getGreen<half3>()
{
    half3 color;
    color.Y = __floats2half2_rn(0.0f, 0.0f);
    color.UV = __floats2half2_rn(1.0f, 0.0f);
    return color;
}

template<typename color_t>
__global__ void initializeMapKernel(color_t* devColor, ushort3* map,
    int imgHeight, int imgWidth)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int idx = ty * imgWidth + tx;

    bool isGreen = (devColor[idx] == getGreen<color_t>());

    map[idx] = make_ushort3(tx, ty, isGreen ? (imgWidth + imgHeight) : 0);
}

__global__ void computeNearestKernel(const ushort3* map_read, ushort3* map_write, int imgHeight, int imgWidth, int* change)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    ushort3 pos = map_read[threadId];
    map_write[threadId] = pos;

    if (pos.z > 0)
    {
        for (int dx = MAX(tx - 1, 0); dx < MIN(tx + 2, imgWidth); ++dx)
        {
            for (int dy = MAX(ty - 1, 0); dy < MIN(ty + 2, imgHeight); ++dy)
            {
                if (abs(tx - dx) + abs(ty - dy) == 1)
                {
                    int neighborIdx = dy * imgWidth + dx;
                    ushort3 path = map_read[neighborIdx];
                    if (path.z + (ushort)1 < pos.z)
                    {
                        map_write[threadId] = make_ushort3(path.x, path.y, path.z + (ushort)1);
                        atomicOr(change, 1);
                    }
                }
            }
        }
    }
}

template<typename color_t>
__global__ void inpaintKernel(color_t* devColor, ushort3* map, int imgHeight, int imgWidth)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int idx = ty * imgWidth + tx;

    if (devColor[idx] == getGreen<color_t>())
    {
        ushort3 pix = map[idx];
        int srcIdx = pix.y * imgWidth + pix.x;

        devColor[idx] = devColor[srcIdx];
    }
}

template<typename color_t>
void inpaintImg(color_t*& devColor, cv::Size dstSize, cudaStream_t& stream, ushort3*& devMap,
    ushort3*& devMap_swap, int*& devChange)
{
    int imgHeight = dstSize.height;
    int imgWidth = dstSize.width;

    int hostChange = 1;

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    initializeMapKernel<color_t><<<gridDim, blockDim, 0, stream>>>(devColor, devMap, imgHeight, imgWidth);

    ushort3* map_read = devMap;
    ushort3* map_write = devMap_swap;

    while (hostChange)
    {
        hostChange = 0;

        cudaMemcpyAsync(devChange, &hostChange, sizeof(int), cudaMemcpyHostToDevice, stream);
        computeNearestKernel <<<gridDim, blockDim, 0, stream>>> (map_read, map_write, imgHeight, imgWidth, devChange);
        cudaMemcpyAsync(&hostChange, devChange, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        ushort3* temp = map_read;
        map_read = map_write;
        map_write = temp;
    }

    inpaintKernel<color_t><<<gridDim, blockDim, 0, stream>>>(devColor, map_read, imgHeight, imgWidth);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template void inpaintImg<float3>(float3*& devColor, cv::Size dstSize, cudaStream_t& stream, ushort3*& devMap,
    ushort3*& devMap_swap, int*& devChange);
template void inpaintImg<double3>(double3*& devColor, cv::Size dstSize, cudaStream_t& stream, ushort3*& devMap,
    ushort3*& devMap_swap, int*& devChange);
template void inpaintImg<half3>(half3*& devColor, cv::Size dstSize, cudaStream_t& stream, ushort3*& devMap,
    ushort3*& devMap_swap, int*& devChange);
