#include "blend_img.cuh"

/**
* Auteur : Enzo DI MARIA
*/

/**
 * @brief A structure to hold constant parameters used in the blending process.
 */
struct ConstantParams
{
    float blendingExponent;
    int imgHeight;
    int imgWidth;
    int numImages;
    float3 emptyColor;
};

/**
 * @brief Holds the constant parameters used during the blending process.
 */
__constant__ ConstantParams constParams;

/**
 * @brief Overloads the multiplication operator for a float3 structure with a scalar.
 *
 * @param a Scalar value.
 * @param b float3 structure.
 * @return A new float3 structure with each component multiplied by the scalar.
 */
__device__ float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

/**
 * @brief Overloads the addition operator for two float3 structures.
 *
 * @param a First float3 structure.
 * @param b Second float3 structure.
 * @return A new float3 structure representing the component-wise sum of the inputs.
 */
__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Overloads the division operator for a float3 structure by a scalar.
 *
 * @param a float3 structure.
 * @param b Scalar value.
 * @return A new float3 structure with each component divided by the scalar.
 */
__device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

/**
 * @brief CUDA kernel that blends multiple images together based on depth and validity information.
 *
 * @param blendedColor Pointer to the blended color input.
 * @param blendedValidity Pointer to the blended validity input.
 * @param blendedDepth Pointer to the blended depth input.
 * @param devColor Pointer to the current image color input.
 * @param devValidity Pointer to the current image validity input.
 * @param devDepth Pointer to the current image depth input.
 * @param outputColor Pointer to the output color buffer.
 * @param outputValidity Pointer to the output validity buffer.
 * @param outputDepth Pointer to the output depth buffer.
 */
__global__ void blendImagesKernel
(
    float3* blendedColor, float* blendedValidity, float* blendedDepth,
    float3* devColor, float* devValidity, float* devDepth,
    float3* outputColor, float* outputValidity, float* outputDepth
)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= constParams.imgWidth || ty >= constParams.imgHeight)
        return;

    int threadId = ty * constParams.imgWidth + tx;

    float sumWeights = 0.0f;
    float inpaintedDepthSumWeights = 0.0f;
    float3 inpaintedDepthColor = make_float3(0, 0, 0);
    float3 finalColor = make_float3(0, 0, 0);

    float quality[2] =
    {
        blendedValidity[threadId] / blendedDepth[threadId],
        devValidity[threadId] / devDepth[threadId]
    };
    bool isValidDepth[2] =
    {
            quality[0] > 0 && !(blendedDepth[threadId] > 0.0f),
            quality[1] > 0 && !(devDepth[threadId] > 0.0f)
    };
    float3 color[2] =
    {
        blendedColor[threadId],
        devColor[threadId]
    };

    for (int i = 0; i < constParams.numImages; i++)
    {
        if (quality[i] > 0)
        {
            if (constParams.blendingExponent > 1.0f)
                quality[i] = powf(quality[i], constParams.blendingExponent);

            if (isValidDepth[i])
            {
                sumWeights += quality[i];
                finalColor = finalColor + (quality[i] * color[i]);
            }
            else
            {
                inpaintedDepthSumWeights += quality[i];
                inpaintedDepthColor = inpaintedDepthColor + (quality[i] * color[i]);
            }
        }
    }

    outputDepth[threadId] = (quality[0] > quality[1]) ? blendedDepth[threadId] : devDepth[threadId];
    outputValidity[threadId] = (blendedValidity[threadId] > devValidity[threadId]) ? blendedValidity[threadId] : devValidity[threadId];

    if (sumWeights == 0)
    {
        if (inpaintedDepthSumWeights == 0.0f)
        {
            outputColor[threadId] = constParams.emptyColor;
        }
        else
        {
            outputColor[threadId] = inpaintedDepthColor / inpaintedDepthSumWeights;
        }
    }
    else
    {
        outputColor[threadId] = finalColor / sumWeights;
    }
}

/**
 * @brief Blends two images together using CUDA based on their depth and validity information.
 *
 * @param devBlendedColor Pointer to the device memory containing the first image's color data.
 * @param devBlendedValidity Pointer to the device memory containing the first image's validity data.
 * @param devBlendedDepth Pointer to the device memory containing the first image's depth data.
 * @param devColor Pointer to the device memory containing the second image's color data.
 * @param devValidity Pointer to the device memory containing the second image's validity data.
 * @param devDepth Pointer to the device memory containing the second image's depth data.
 * @param outputSize The size of the output image.
 * @param emptyColor The color used for empty pixels (output when no valid color is found).
 * @param devOutputColor Pointer to the device memory where the output color data will be stored.
 * @param devOutputValidity Pointer to the device memory where the output validity data will be stored.
 * @param devOutputDepth Pointer to the device memory where the output depth data will be stored.
 * @param blendingExponent The exponent used to adjust the blending quality.
 *
 * @throws std::runtime_error If a CUDA error occurs.
 */
void blendImages
(
    float3* devBlendedColor, float* devBlendedValidity, float* devBlendedDepth,
    float3* devColor, float* devValidity, float* devDepth, cv::Size outputSize,
    cv::Vec3f emptyColor, float3*& devOutputColor, float*& devOutputValidity, float*& devOutputDepth, float blendingExponent
)
{
    int imgHeight = outputSize.height;
    int imgWidth = outputSize.width;
    int imgSize = imgWidth * imgHeight;
    int numImages = 2;

    float3 emptyColor3 = make_float3(emptyColor[0], emptyColor[1], emptyColor[2]);

    ConstantParams params = { blendingExponent, imgHeight, imgWidth, numImages, emptyColor3 };
    cudaMemcpyToSymbol(constParams, &params, sizeof(ConstantParams));

    cudaMalloc(&devOutputColor, imgSize * sizeof(float3));
    cudaMalloc(&devOutputValidity, imgSize * sizeof(float));
    cudaMalloc(&devOutputDepth, imgSize * sizeof(float));

    int blockWidth = 8;
    int blockHeight = 16;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    blendImagesKernel <<<gridDim, blockDim>>> (
        devBlendedColor, devBlendedValidity, devBlendedDepth,
        devColor, devValidity, devDepth,
        devOutputColor, devOutputValidity, devOutputDepth);

    cudaDeviceSynchronize();

    cudaFree(devBlendedColor);
    cudaFree(devBlendedValidity);
    cudaFree(devBlendedDepth);

    cudaFree(devColor);
    cudaFree(devValidity);
    cudaFree(devDepth);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}
