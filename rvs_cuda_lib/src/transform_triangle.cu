#include "transform_triangle.cuh"
#include <iostream>

/**
* Auteur : Enzo DI MARIA
*/

/**
 * @brief A constant structure to hold the dimensions of the image (or the grid) being processed.
 */
__constant__ int imgDim[2];

/**
* @brief  A constant structure to hold the desired size of the output color and depth images.
*/
__constant__ int outputImgDim[2];

/**
 * @brief Computes the squared Euclidean distance between two points in 2D space.
 *
 * @param p1 First point as a float2 containing the x and y coordinates.
 * @param p2 Second point as a float2 containing the x and y coordinates.
 *
 * @return The squared distance between p1 and p2.
 */
__device__ inline float dot(float2 p1, float2 p2)
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

/**
 * @brief Evaluates the validity of a triangle based on the lengths of its sides.
 *
 * @param posA The position of the first vertex of the triangle.
 * @param posB The position of the second vertex of the triangle.
 * @param posC The position of the third vertex of the triangle.
 *
 * @return A quality score for the triangle, ranging from 1 to 10000.
 */
__device__ inline float isTriValid(float2 posA, float2 posB, float2 posC)
{
    float ab = dot(posA, posB);
    float ac = dot(posA, posC);
    float bc = dot(posB, posC);

    float quality = 10000.f - 1000.f * sqrtf(fmaxf(bc, fmaxf(ab, ac)));

    return fminf(10000.f, fmaxf(1.f, quality));
}

struct DepthAndValidity
{
    float depth;
    float validity;
};

union DepthAndValidityUnion
{
    DepthAndValidity dv;
    unsigned long long int encoded;
};

__device__ unsigned long long int encodeDepthAndValidity(float depth, float validity)
{
    DepthAndValidityUnion u;
    u.dv.depth = depth;
    u.dv.validity = validity;
    return u.encoded;
}

__device__ DepthAndValidity decodeDepthAndValidity(unsigned long long int encoded)
{
    DepthAndValidityUnion u;
    u.encoded = encoded;
    return u.dv;
}

/**
 * @brief Processes a single pixel within a triangle, updating color and depth based on barycentric coordinates.
 *
 * @param px The x-coordinate of the pixel to process.
 * @param py The y-coordinate of the pixel to process.
 * @param invArea The precomputed inverse area of the triangle.
 * @param outputColor The global result buffer where the new color will be stored.
 * @param outputDepth The global buffer where the new depth will be stored.
 * @param quality The global buffer indicating the shape quality of the triangles.
 * @param validity The global buffer indicating the shape quality of the triangles.
 * @param triangleValidity The threshold for the validity of triangle shape updates.
 * @param posA The position of the first vertex of the triangle.
 * @param posB The position of the second vertex of the triangle.
 * @param posC The position of the third vertex of the triangle.
 * @param colA The color of the first vertex of the triangle.
 * @param colB The color of the second vertex of the triangle.
 * @param colC The color of the third vertex of the triangle.
 * @param depA The depth of the first vertex of the triangle.
 * @param depB The depth of the second vertex of the triangle.
 * @param depC The depth of the third vertex of the triangle.
 */
__device__ void processPixel
(
    int px, int py, float invArea, float3* outputColor, unsigned long long int* outputDepthValidity, float triangleValidity,
    float2 posA, float2 posB, float2 posC, float3 colA, float3 colB, float3 colC, float depA, float depB, float depC
)
{
    int pixelId = py * outputImgDim[0] + px;

    float offsetPx = (float)px + 0.5f;
    float offsetPy = (float)py + 0.5f;

    float lambda1 = invArea * ((posB.y - posC.y) * (offsetPx - posC.x) + (posC.x - posB.x) * (offsetPy - posC.y));
    if (lambda1 < 0) return;

    float lambda2 = invArea * ((posC.y - posA.y) * (offsetPx - posC.x) + (posA.x - posC.x) * (offsetPy - posC.y));
    if (lambda2 < 0) return;

    float lambda3 = 1.0f - lambda1 - lambda2;
    if (lambda3 < 0) return;

    float depth = fmaf(depA, lambda1, fmaf(depB, lambda2, depC * lambda3));

    unsigned long long int newEncoded = encodeDepthAndValidity(depth, triangleValidity);
    unsigned long long int oldEncoded = atomicExch(&outputDepthValidity[pixelId], newEncoded);

    DepthAndValidity oldDV = decodeDepthAndValidity(oldEncoded);

    float ratio = depth / oldDV.depth;

    if (oldDV.validity * ratio * ratio * ratio > triangleValidity)
    {
        atomicExch(&outputDepthValidity[pixelId], oldEncoded);
        return;
    }

    float3 newColor = make_float3
    (
        fmaf(colA.x, lambda1, fmaf(colB.x, lambda2, colC.x * lambda3)),
        fmaf(colA.y, lambda1, fmaf(colB.y, lambda2, colC.y * lambda3)),
        fmaf(colA.z, lambda1, fmaf(colB.z, lambda2, colC.z * lambda3))
    );

    oldEncoded = atomicExch(&outputDepthValidity[pixelId], newEncoded);
    oldDV = decodeDepthAndValidity(oldEncoded);
    ratio = depth / oldDV.depth;

    if (oldDV.validity * ratio * ratio * ratio > triangleValidity)
    {
        atomicExch(&outputDepthValidity[pixelId], oldEncoded);
        return;
    }

    outputColor[pixelId] = newColor;
}

/**
 * @brief Processes a triangle to update colors and depths of its pixels.
 *
 * @param posA The position of the first vertex of the triangle.
 * @param posB The position of the second vertex of the triangle.
 * @param posC The position of the third vertex of the triangle.
 * @param colA The color of the first vertex of the triangle.
 * @param colB The color of the second vertex of the triangle.
 * @param colC The color of the third vertex of the triangle.
 * @param depA The depth of the first vertex of the triangle.
 * @param depB The depth of the second vertex of the triangle.
 * @param depC The depth of the third vertex of the triangle.
 * @param outputColor The global result buffer where the new color will be stored.
 * @param outputDepthValidity The global buffer where the new depth and validity will be stored.
 */
__device__ void processTriangle
(
    float2 posA, float2 posB, float2 posC, float3 colA, float3 colB, float3 colC, float depA, float depB, float depC, float3* outputColor, unsigned long long int* outputDepthValidity
)
{
    if (depA <= 0 || posA.x <= 0 || isnan(depA) || isnan(posA.x))
        return;

    int xMin = fmaxf(0, floorf(fminf(fminf(posA.x, posB.x), posC.x)));
    int xMax = fminf(outputImgDim[0] - 1, ceilf(fmaxf(fmaxf(posA.x, posB.x), posC.x)));
    int yMin = fmaxf(0, floorf(fminf(fminf(posA.y, posB.y), posC.y)));
    int yMax = fminf(outputImgDim[1] - 1, ceilf(fmaxf(fmaxf(posA.y, posB.y), posC.y)));

    if (yMin >= yMax || xMin >= xMax)
        return;

    float invArea = (posB.y - posC.y) * (posA.x - posC.x) + (posC.x - posB.x) * (posA.y - posC.y);

    if (invArea <= 0.0f)
        return;

    invArea = 1.0f / invArea;

    float triangleValidity = isTriValid(posA, posB, posC);

    for (int dy = yMin; dy <= yMax; ++dy)
        for (int dx = xMin; dx <= xMax; ++dx)
            processPixel(dx, dy, invArea, outputColor, outputDepthValidity, triangleValidity, posA, posB, posC, colA, colB, colC, depA, depB, depC);
}

/**
 * @brief CUDA kernel that colorizes triangles based on depth and position data.
 *
 * The kernel assumes that each thread corresponds to a single pixel, indexed by the 2D block and grid
 * dimensions. It uses shared constants 'imgDim' to represent image dimensions and 'isErp' to indicate if
 * equirectangular projection wrapping is required.
 *
 * @param inputColor The input image containing colors in YUV.
 * @param inputDepth The input image containing depth values.
 * @param inputPositions The positions of triangle vertices.
 * @param outputColor The global result buffer where the new color will be stored.
 * @param outputDepthValidity The global buffer where the new depth and validity will be stored.
 * @param horizontalWarping A boolean constant indicating if the image has a horizontal warping.
 */
__global__ void colorizeTriangleKernel(float3* inputColor, float* inputDepth, float2* inputPositions, float3* outputColor, unsigned long long int* outputDepthValidity, bool horizontalWarping)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int threadId = ty * imgDim[0] + tx;

    if (tx >= imgDim[0] - 1 || ty >= imgDim[1] - 1)
        return;

    int3 triPixelIds = make_int3(threadId + 1 + imgDim[0], threadId + 1, threadId + imgDim[0]);

    float depB = inputDepth[triPixelIds.y];
    if (isnan(depB) || depB <= 0.f)
        return;
    float depC = inputDepth[triPixelIds.z];
    if (isnan(depC) || depC <= 0.f)
        return;

    float2 posB = inputPositions[triPixelIds.y];
    if (isnan(posB.x) || posB.x <= 0.f)
        return;
    float2 posC = inputPositions[triPixelIds.z];
    if (isnan(posC.x) || posC.x <= 0.f)
        return;

    float3 colB = inputColor[triPixelIds.y];
    float3 colC = inputColor[triPixelIds.z];

    processTriangle(inputPositions[threadId], posB, posC, inputColor[threadId], colB, colC, inputDepth[threadId], depB, depC, outputColor, outputDepthValidity);
    processTriangle(inputPositions[triPixelIds.x], posC, posB, inputColor[triPixelIds.x], colC, colB, inputDepth[triPixelIds.x], depC, depB, outputColor, outputDepthValidity);

    if (horizontalWarping && tx == 0)
    {
        int4 endTriPixelIds = make_int4(ty * imgDim[0] + imgDim[0] - 1, ty * imgDim[0], (ty + 1) * imgDim[0] + imgDim[0] - 1, (ty + 1) * imgDim[0]);

        colB = inputColor[endTriPixelIds.y];
        colC = inputColor[endTriPixelIds.z];

        depB = inputDepth[endTriPixelIds.y];
        depC = inputDepth[endTriPixelIds.z];

        posB = inputPositions[endTriPixelIds.y];
        posC = inputPositions[endTriPixelIds.z];

        processTriangle(inputPositions[endTriPixelIds.x], posB, posC, inputColor[endTriPixelIds.x], colB, colC, inputDepth[endTriPixelIds.x], depB, depC, outputColor, outputDepthValidity);
        processTriangle(inputPositions[endTriPixelIds.w], posC, posB, inputColor[endTriPixelIds.w], colC, colB, inputDepth[endTriPixelIds.w], depC, depB, outputColor, outputDepthValidity);
    }
}

__global__ void initializeArrayWithEncodedValues(unsigned long long int* array, float depth, float validity, int size)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < size)
    {
        array[threadId] = encodeDepthAndValidity(depth, validity);
    }
}

__global__ void separateDepthAndValidityKernel(unsigned long long int* outputDepthValidity, float* outputDepth, float* validity, int size)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < size)
    {
        DepthAndValidity dv = decodeDepthAndValidity(outputDepthValidity[threadId]);

        outputDepth[threadId] = dv.depth;
        validity[threadId] = dv.validity;
    }
}


/**
 * @brief Transforms triangles from the input image to the output image based on depth and position information.
 *
 * @param inputColor Input color image as a 3-channel float matrix containing the YUV values.
 * @param inputDepth Input depth image as a single-channel float matrix containing depth values.
 * @param inputPositions Input positions image as a 2-channel float matrix containing the x and y coordinates.
 * @param outputSize The desired size of the output color and depth images.
 * @param baseDepth Base depth image as a single-channel float matrix.
 * @param baseValidity Base validity mask as a single-channel float matrix.
 * @param horizontalWrap A boolean flag indicating whether to process the image for horizontal wrapping.
 * @param devOutputColor Pointer to the output color buffer.
 * @param devOutputDepth Pointer to the output depth buffer.
 * @param devQuality Pointer to the quality buffer.
 * @param devValidity Pointer to the validity buffer.
 */
void transform_trianglesMethod
(
    float3* devInputColor, cv::Size inputSize, float* devInputDepth, float2* devInputPositions, cv::Size outputSize, bool horizontalWrap,
    float3*& devOutputColor, float*& devOutputDepth, float*& devValidity
)
{
    int imgWidth = inputSize.width;
    int imgHeight = inputSize.height;

    int outputImgWidth = outputSize.width;
    int outputImgHeight = outputSize.height;
    int outputImgSize = outputImgWidth * outputImgHeight;

    int hostImgDim[2] = { imgWidth, imgHeight };
    int hostOutputSize[2] = { outputImgWidth, outputImgHeight };

    unsigned long long int* devOutputDepthValidity;

    cudaMalloc(&devOutputDepthValidity, outputImgSize * sizeof(unsigned long long int));
    cudaMalloc(&devOutputColor, outputImgSize * sizeof(float3));
    cudaMalloc(&devOutputDepth, outputImgSize * sizeof(float));
    cudaMalloc(&devValidity, outputImgSize * sizeof(float));

    int blockSize = 256;
    int numBlocks = (outputImgSize + blockSize - 1) / blockSize;

    initializeArrayWithEncodedValues <<<numBlocks, blockSize>>> (devOutputDepthValidity, INFINITY, 0, outputImgSize);

    cudaMemcpyToSymbol(imgDim, &hostImgDim, 2 * sizeof(int));
    cudaMemcpyToSymbol(outputImgDim, &hostOutputSize, 2 * sizeof(int));

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    colorizeTriangleKernel <<<gridDim, blockDim>>> (devInputColor, devInputDepth, devInputPositions, devOutputColor, devOutputDepthValidity, horizontalWrap);

    cudaDeviceSynchronize();

    separateDepthAndValidityKernel <<<numBlocks, blockSize>>> (devOutputDepthValidity, devOutputDepth, devValidity, outputImgSize);

    cudaDeviceSynchronize();

    cudaFree(devInputColor);
    cudaFree(devInputDepth);
    cudaFree(devInputPositions);
    cudaFree(devOutputDepthValidity);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}
