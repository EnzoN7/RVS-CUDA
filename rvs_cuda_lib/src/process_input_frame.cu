#include "process_input_frame.cuh"
#include <iostream>
#include <opencv2/imgproc.hpp>

/**
* Auteur : Enzo DI MARIA
*/

/**
 * @brief CUDA kernel that resizes UV components of an image using bilinear interpolation.
 *
 * @param uInput Pointer to the U component input data.
 * @param vInput Pointer to the V component input data.
 * @param uOutput Pointer to the resized U component output data.
 * @param vOutput Pointer to the resized V component output data.
 * @param inputWidth Width of the input UV image.
 * @param inputHeight Height of the input UV image.
 * @param outputWidth Width of the output UV image.
 * @param outputHeight Height of the output UV image.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the input data.
 */
__global__ void resizeUV(void* uInput, void* vInput, float* uOutput, float* vOutput, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int type)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight)
    {
        float scaleX = (float)(inputWidth) / outputWidth;
        float scaleY = (float)(inputHeight) / outputHeight;

        float srcX = x * scaleX;
        float srcY = y * scaleY;

        int x1 = (int)srcX;
        int y1 = (int)srcY;
        int x2 = fminf(x1 + 1, inputWidth - 1);
        int y2 = fminf(y1 + 1, inputHeight - 1);

        float dx = srcX - x1;
        float dy = srcY - y1;

        float val1, val2;

        if (type == CV_32F)
        {
            val1 = ((float*)uInput)[y1 * inputWidth + x1] * (1 - dx) + ((float*)uInput)[y1 * inputWidth + x2] * dx;
            val2 = ((float*)uInput)[y2 * inputWidth + x1] * (1 - dx) + ((float*)uInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_16U)
        {
            val1 = ((unsigned short*)uInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned short*)uInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned short*)uInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned short*)uInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_8U)
        {
            val1 = ((unsigned char*)uInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned char*)uInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned char*)uInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned char*)uInput)[y2 * inputWidth + x2] * dx;
        }

        uOutput[y * outputWidth + x] = val1 * (1 - dy) + val2 * dy;

        if (type == CV_32F)
        {
            val1 = ((float*)vInput)[y1 * inputWidth + x1] * (1 - dx) + ((float*)vInput)[y1 * inputWidth + x2] * dx;
            val2 = ((float*)vInput)[y2 * inputWidth + x1] * (1 - dx) + ((float*)vInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_16U)
        {
            val1 = ((unsigned short*)vInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned short*)vInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned short*)vInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned short*)vInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_8U)
        {
            val1 = ((unsigned char*)vInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned char*)vInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned char*)vInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned char*)vInput)[y2 * inputWidth + x2] * dx;
        }

        vOutput[y * outputWidth + x] = val1 * (1 - dy) + val2 * dy;
    }
}

/**
 * @brief Resizes the UV channels of an image using CUDA and stores the output on the device.
 *
 * @param inputs Array of cv::Mat containing the input U and V channels.
 * @param devUOutput Pointer to the device memory where the resized U channel will be stored.
 * @param devVOutput Pointer to the device memory where the resized V channel will be stored.
 * @param size The desired output size of the UV channels.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the input data.
 *
 * @throws std::invalid_argument If the input type is not supported.
 * @throws std::runtime_error If a CUDA error occurs.
 */
void resizeImage(cv::Mat inputs[], float*& devUOutput, float*& devVOutput, cv::Size size, int type)
{
    void* devUInput;
    void* devVInput;

    size_t inputSize = inputs[0].total() * inputs[0].elemSize();
    size_t outputSize = size.area() * sizeof(float);

    cudaMalloc(&devUInput, inputSize);
    cudaMalloc(&devVInput, inputSize);
    cudaMalloc(&devUOutput, outputSize);
    cudaMalloc(&devVOutput, outputSize);

    cudaMemcpy(devUInput, inputs[0].data, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devVInput, inputs[1].data, inputSize, cudaMemcpyHostToDevice);

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((size.width - 1 + blockWidth) / blockWidth, (size.height - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    if (type == CV_32F || type == CV_16U || type == CV_8U)
    {
        resizeUV<<<gridDim, blockDim>>>(devUInput, devVInput, devUOutput, devVOutput, inputs[0].cols, inputs[0].rows, size.width, size.height, type);
    }
    else
    {
        cudaFree(devUInput);
        cudaFree(devVInput);
        cudaFree(devUOutput);
        cudaFree(devVOutput);
        inputs[0].release();
        inputs[1].release();

        throw std::invalid_argument("Unsupported CV type");
    }

    cudaDeviceSynchronize();

    cudaFree(devUInput);
    cudaFree(devVInput);
    inputs[0].release();
    inputs[1].release();

    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(state));
    }
}

/**
 * @brief CUDA kernel that resizes the Y component of an image using bilinear interpolation.
 *
 * @param yInput Pointer to the Y component input data.
 * @param yOutput Pointer to the resized Y component output data.
 * @param inputWidth Width of the input Y image.
 * @param inputHeight Height of the input Y image.
 * @param outputWidth Width of the output Y image.
 * @param outputHeight Height of the output Y image.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the input data.
 */
__global__ void resizeY(void* yInput, float* yOutput, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int type)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight)
    {
        float scaleX = (float)(inputWidth) / outputWidth;
        float scaleY = (float)(inputHeight) / outputHeight;

        float srcX = x * scaleX;
        float srcY = y * scaleY;

        int x1 = (int)srcX;
        int y1 = (int)srcY;
        int x2 = fminf(x1 + 1, inputWidth - 1);
        int y2 = fminf(y1 + 1, inputHeight - 1);

        float dx = srcX - x1;
        float dy = srcY - y1;

        float val1, val2;
        if (type == CV_32F)
        {
            val1 = ((float*)yInput)[y1 * inputWidth + x1] * (1 - dx) + ((float*)yInput)[y1 * inputWidth + x2] * dx;
            val2 = ((float*)yInput)[y2 * inputWidth + x1] * (1 - dx) + ((float*)yInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_16U)
        {
            val1 = ((unsigned short*)yInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned short*)yInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned short*)yInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned short*)yInput)[y2 * inputWidth + x2] * dx;
        }
        else if (type == CV_8U)
        {
            val1 = ((unsigned char*)yInput)[y1 * inputWidth + x1] * (1 - dx) + ((unsigned char*)yInput)[y1 * inputWidth + x2] * dx;
            val2 = ((unsigned char*)yInput)[y2 * inputWidth + x1] * (1 - dx) + ((unsigned char*)yInput)[y2 * inputWidth + x2] * dx;
        }

        yOutput[y * outputWidth + x] = val1 * (1 - dy) + val2 * dy;
    }
}

/**
 * @brief CUDA kernel that merges the Y, U, and V channels into a single float3 buffer.
 *
 * @param yChannel Pointer to the Y component data.
 * @param uChannel Pointer to the U component data.
 * @param vChannel Pointer to the V component data.
 * @param yuvChannels Pointer to the output float3 buffer containing the merged YUV channels.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param colorScale The scaling factor applied to the color values.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the Y component.
 */
__global__ void mergeYUV(const void* yChannel, const float* uChannel, const float* vChannel, float3* yuvChannels, int width, int height, float colorScale, int type)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        float3 pixel;

        if (type == CV_32F)
        {
            pixel.x = ((float*)yChannel)[idx] * colorScale;
        }
        else if (type == CV_16U)
        {
            pixel.x = ((unsigned short*)yChannel)[idx] * colorScale;
        }
        else if (type == CV_8U)
        {
            pixel.x = ((unsigned char*)yChannel)[idx] * colorScale;
        }
        pixel.y = uChannel[idx] * colorScale;
        pixel.z = vChannel[idx] * colorScale;

        yuvChannels[idx] = pixel;
    }
}

/**
 * @brief Merges the Y, U, and V channels of an image and stores the output on the device.
 *
 * @param yInput The input cv::Mat containing the Y channel.
 * @param devU Pointer to the device memory containing the U channel.
 * @param devV Pointer to the device memory containing the V channel.
 * @param devYUV Pointer to the device memory where the merged YUV data will be stored.
 * @param outputSize The desired output size.
 * @param yType The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the Y channel.
 * @param colorScale The scaling factor applied to the color values.
 *
 * @throws std::invalid_argument If the input type is not supported.
 * @throws std::runtime_error If a CUDA error occurs.
 */
void mergeFrame(cv::Mat& yInput, float* devU, float* devV, float3*& devYUV, cv::Size outputSize, int yType, float colorScale)
{
    int width = outputSize.width;
    int height = outputSize.height;

    void* devY;

    size_t yInputSize = yInput.total() * yInput.elemSize();
    size_t uvInputSize = outputSize.area() * sizeof(float);
    size_t yuvOutputSize = outputSize.area() * sizeof(float3);

    cudaMalloc(&devY, yInputSize);
    cudaMalloc(&devYUV, yuvOutputSize);

    cudaMemcpy(devY, yInput.data, yInputSize, cudaMemcpyHostToDevice);

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((width - 1 + blockWidth) / blockWidth, (height - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    if (yInput.size() != outputSize)
    {
        float* devRescaledY;
        cudaMalloc(&devRescaledY, outputSize.area() * sizeof(float));

        if (yType == CV_32F || yType == CV_16U || yType == CV_8U)
        {
            resizeY<<<gridDim, blockDim>>>(devY, devRescaledY, yInput.cols, yInput.rows, outputSize.width, outputSize.height, yType);
        }
        else
        {
            cudaFree(devY);
            cudaFree(devU);
            cudaFree(devV);
            cudaFree(devRescaledY);
            cudaFree(devYUV);
            throw std::invalid_argument("Unsupported CV type");
        }

        cudaDeviceSynchronize();

        mergeYUV<<<gridDim, blockDim>>>(devRescaledY, devU, devV, devYUV, width, height, colorScale, CV_32F);

        cudaDeviceSynchronize();

        cudaFree(devRescaledY);
    }
    else
    {
        if (yType == CV_32F || yType == CV_16U || yType == CV_8U)
        {
            mergeYUV<<<gridDim, blockDim>>>(devY, devU, devV, devYUV, width, height, colorScale, yType);
        }
        else
        {
            cudaFree(devY);
            cudaFree(devU);
            cudaFree(devV);
            cudaFree(devYUV);
            yInput.release();
            throw std::invalid_argument("Unsupported CV type");
        }

        cudaDeviceSynchronize();
    }

    cudaFree(devY);
    cudaFree(devU);
    cudaFree(devV);
    yInput.release();

    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(state));
    }
}

/**
 * @brief CUDA kernel that normalizes and converts depth values based on near and far plane distances.
 *
 * @param src Pointer to the input depth data.
 * @param dst Pointer to the output normalized depth data.
 * @param scale Scaling factor for the depth values.
 * @param near Near plane distance.
 * @param far Far plane distance.
 * @param rows Number of rows in the image.
 * @param cols Number of columns in the image.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the input data.
 * @param hasInvalidDepth Whether the depth data contains invalid values that should be set to NaN.
 */
__global__ void normalize(void* src, float* dst, float scale, float near, float far, int rows, int cols, int type, bool hasInvalidDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int idx = y * cols + x;
        float depth = 0.0f;

        switch (type)
        {
            case CV_32F:
                depth = ((float*)src)[idx] * scale;
                break;
            case CV_16U:
                depth = ((unsigned short*)src)[idx] * scale;
                break;
            case CV_8U:
                depth = ((unsigned char*)src)[idx] * scale;
                break;
        }

        if (far >= 1000.f)
        {
            depth = near / depth;
        }
        else
        {
            depth = far * near / (near + depth * (far - near));
        }

        if (hasInvalidDepth && depth == 0)
        {
            depth = NAN;
        }

        dst[idx] = depth;
    }
}

/**
 * @brief Normalizes and resizes depth data, handling different OpenCV data types and storing the output on the device.
 *
 * @param src The input cv::Mat containing the depth data.
 * @param devDst Pointer to the device memory where the normalized depth data will be stored.
 * @param size The desired output size.
 * @param scale Scaling factor for the depth values.
 * @param near Near plane distance.
 * @param far Far plane distance.
 * @param hasInvalidDepth Whether the depth data contains invalid values that should be set to NaN.
 * @param type The OpenCV data type (e.g., CV_32F, CV_16U, CV_8U) of the input data.
 *
 * @throws std::invalid_argument If the input type is not supported.
 * @throws std::runtime_error If a CUDA error occurs.
 */
void formatDepth(cv::Mat& src, float*& devDst, cv::Size size, float scale, float near, float far, bool hasInvalidDepth, int type)
{
    int rows = size.height;
    int cols = size.width;

    size_t inputSize = src.total() * src.elemSize();
    size_t outputSize = size.area() * sizeof(float);

    void* devSrc;

    cudaMalloc(&devSrc, inputSize);
    cudaMalloc(&devDst, outputSize);

    cudaMemcpy(devSrc, src.data, inputSize, cudaMemcpyHostToDevice);

    int blockWidth = 16;
    int blockHeight = 8;
    dim3 gridDim((cols - 1 + blockWidth) / blockWidth, (rows - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    if (src.size() != size)
    {
        float* devSrcRescaled;

        cudaMalloc(&devSrcRescaled, outputSize);

        if (type == CV_32F || type == CV_16U || type == CV_8U)
        {
            resizeY<<<gridDim, blockDim>>>(devSrc, devSrcRescaled, src.cols, src.rows, cols, rows, type);
        }
        else
        {
            cudaFree(devSrc);
            cudaFree(devDst);
            cudaFree(devSrcRescaled);
            throw std::invalid_argument("Unsupported CV type");
        }

        cudaDeviceSynchronize();

        normalize<<<gridDim, blockDim>>>(devSrcRescaled, devDst, scale, near, far, rows, cols, CV_32F, hasInvalidDepth);
    }
    else
    {
        if (type == CV_32F || type == CV_16U || type == CV_8U)
        {
            normalize<<<gridDim, blockDim>>>(devSrc, devDst, scale, near, far, rows, cols, type, hasInvalidDepth);
        }
        else
        {
            cudaFree(devSrc);
            cudaFree(devDst);
            throw std::invalid_argument("Unsupported CV type");
        }
    }

    cudaDeviceSynchronize();

    cudaFree(devSrc);

    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(state));
    }
}

/**
 * @brief CUDA kernel that separates Y, U, and V channels from a float3 buffer into individual channels.
 *
 * @param src Pointer to the input float3 buffer containing YUV data.
 * @param Y Pointer to the output Y channel buffer.
 * @param U Pointer to the output U channel buffer.
 * @param V Pointer to the output V channel buffer.
 * @param rows Number of rows in the image.
 * @param cols Number of columns in the image.
 */
__global__ void separateChannelsKernel(float3* src, float* Y, float* U, float* V, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * cols + x;

    if (y < rows && x < cols)
    {
        float3 pixel = src[idx];

        Y[idx] = pixel.x;
        U[idx] = pixel.y;
        V[idx] = pixel.z;
    }
}

/**
 * @brief CUDA kernel that performs quantization of float data into different OpenCV types.
 *
 * @param src Pointer to the input float data.
 * @param dst Pointer to the output quantized data (device memory).
 * @param rows Number of rows in the image.
 * @param cols Number of columns in the image.
 * @param cv_depth The target OpenCV data type (e.g., CV_8U, CV_16U, CV_32F).
 * @param max_val The maximum value used for quantization.
 */
__global__ void quantizationKernel(float* src, void* dst, int rows, int cols, int cv_depth, unsigned max_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * cols + x;

    if (y < rows && x < cols)
    {
        float pixel = src[idx];

        switch (cv_depth)
        {
            case CV_8U:
                ((unsigned char*)dst)[idx] = pixel * max_val;
                break;
            case CV_16U:
                ((unsigned short*)dst)[idx] = pixel * max_val;
                break;
            case CV_32F:
                ((float*)dst)[idx] = pixel * max_val;
                break;
        }
    }
}

/**
 * @brief Performs quantization of YUV channels, resizing as needed, and stores the output in the provided cv::Mat.
 *
 * @param devSrc Pointer to the device memory containing the input YUV data as a float3 buffer.
 * @param dstY cv::Mat to store the quantized Y channel.
 * @param dstU cv::Mat to store the quantized U channel.
 * @param dstV cv::Mat to store the quantized V channel.
 * @param rows Number of rows in the input YUV data.
 * @param cols Number of columns in the input YUV data.
 * @param outputSize The desired output size.
 * @param cv_depth The target OpenCV data type (e.g., CV_8U, CV_16U, CV_32F).
 * @param max_val The maximum value used for quantization.
 *
 * @throws std::invalid_argument If the input type is not supported.
 * @throws std::runtime_error If a CUDA error occurs.
 */
void quantization(float3* devSrc, cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV, int rows, int cols, cv::Size outputSize, int cv_depth, unsigned max_val)
{
    size_t dstSizeY, dstSizeUV;
    cv::Size uvSize(outputSize.width / 2, outputSize.height / 2);

    switch (cv_depth)
    {
        case CV_8U:
            dstSizeY = outputSize.height * outputSize.width * sizeof(unsigned char);
            dstSizeUV = uvSize.height * uvSize.width * sizeof(unsigned char);
            break;
        case CV_16U:
            dstSizeY = outputSize.height * outputSize.width * sizeof(unsigned short);
            dstSizeUV = uvSize.height * uvSize.width * sizeof(unsigned short);
            break;
        case CV_32F:
            dstSizeY = outputSize.height * outputSize.width * sizeof(float);
            dstSizeUV = uvSize.height * uvSize.width * sizeof(float);
            break;
        default:
            throw std::invalid_argument("unsupported CV depth");
    }

    float *devY, *devU, *devV;
    cudaMalloc(&devY, rows * cols * sizeof(float));
    cudaMalloc(&devU, rows * cols * sizeof(float));
    cudaMalloc(&devV, rows * cols * sizeof(float));

    int blockWidth = 16;
    int blockHeight = 8;

    dim3 blockDim(blockWidth, blockHeight);

    dim3 gridDim((cols + blockWidth - 1) / blockWidth, (rows + blockHeight - 1) / blockHeight);
    dim3 resizeGridDimUV((uvSize.width + blockWidth - 1) / blockWidth, (uvSize.height + blockHeight - 1) / blockHeight);

    separateChannelsKernel<<<gridDim, blockDim>>>(devSrc, devY, devU, devV, rows, cols);

    cudaDeviceSynchronize();

    void* devDstY;
    void* devDstU;
    void* devDstV;
    cudaMalloc(&devDstY, dstSizeY);
    cudaMalloc(&devDstU, dstSizeUV);
    cudaMalloc(&devDstV, dstSizeUV);

    float* resizedU, * resizedV;

    cudaMalloc(&resizedU, uvSize.height * uvSize.width * sizeof(float));
    cudaMalloc(&resizedV, uvSize.height * uvSize.width * sizeof(float));

    resizeUV<<<resizeGridDimUV, blockDim>>>(devU, devV, resizedU, resizedV, cols, rows, uvSize.width, uvSize.height, CV_32F);

    if (cv::Size(cols, rows) != outputSize)
    {
        float *resizedY;
        cudaMalloc(&resizedY, outputSize.height * outputSize.width * sizeof(float));

        dim3 resizeGridDimY((outputSize.width + blockWidth - 1) / blockWidth, (outputSize.height + blockHeight - 1) / blockHeight);

        resizeY<<<resizeGridDimY, blockDim>>>(devY, resizedY, cols, rows, outputSize.width, outputSize.height, CV_32F);

        cudaDeviceSynchronize();

        quantizationKernel<<<resizeGridDimY, blockDim>>>(resizedY, devDstY, outputSize.height, outputSize.width, cv_depth, max_val);

        cudaFree(resizedY);
    }
    else
    {
        quantizationKernel<<<gridDim, blockDim>>>(devY, devDstY, outputSize.height, outputSize.width, cv_depth, max_val);
    }

    quantizationKernel<<<resizeGridDimUV, blockDim>>>(resizedU, devDstU, uvSize.height, uvSize.width, cv_depth, max_val);
    quantizationKernel<<<resizeGridDimUV, blockDim>>>(resizedV, devDstV, uvSize.height, uvSize.width, cv_depth, max_val);

    cudaFree(resizedU);
    cudaFree(resizedV);

    cudaDeviceSynchronize();

    switch (cv_depth)
    {
        case CV_8U:
            cudaMemcpy(dstY.ptr<unsigned char>(), devDstY, dstSizeY, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstU.ptr<unsigned char>(), devDstU, dstSizeUV, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstV.ptr<unsigned char>(), devDstV, dstSizeUV, cudaMemcpyDeviceToHost);
            break;
        case CV_16U:
            cudaMemcpy(dstY.ptr<unsigned short>(), devDstY, dstSizeY, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstU.ptr<unsigned short>(), devDstU, dstSizeUV, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstV.ptr<unsigned short>(), devDstV, dstSizeUV, cudaMemcpyDeviceToHost);
            break;
        case CV_32F:
            cudaMemcpy(dstY.ptr<float>(), devDstY, dstSizeY, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstU.ptr<float>(), devDstU, dstSizeUV, cudaMemcpyDeviceToHost);
            cudaMemcpy(dstV.ptr<float>(), devDstV, dstSizeUV, cudaMemcpyDeviceToHost);
            break;
    }

    cudaFree(devY);
    cudaFree(devU);
    cudaFree(devV);
    cudaFree(devDstY);
    cudaFree(devDstU);
    cudaFree(devDstV);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}