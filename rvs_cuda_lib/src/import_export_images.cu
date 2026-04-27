/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
    Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "import_export_images.cuh"


template<typename channel_read_t, typename channel_t>
__device__ inline channel_t bilinearInterpolate(const channel_read_t* src, int width, int height, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, float dx, float dy)
{
    if constexpr (std::is_same<channel_t, half>::value)
    {
        half2 Hdx = __floats2half2_rn(dx, 1.0f - dx);
        half2 Hdy = __floats2half2_rn(dy, 1.0f - dy);

        half val1 = __float2half(static_cast<float>(src[y1 * width + x1])) * Hdx.y +
                    __float2half(static_cast<float>(src[y1 * width + x2])) * Hdx.x;

        half val2 = __float2half(static_cast<float>(src[y2 * width + x1])) * Hdx.y +
                    __float2half(static_cast<float>(src[y2 * width + x2])) * Hdx.x;

        return val1 * Hdy.y + val2 * Hdy.x;
    }
    else
    {
        channel_t val1 = src[y1 * width + x1] * ((channel_t)1.0 - dx) + src[y1 * width + x2] * dx;
        channel_t val2 = src[y2 * width + x1] * ((channel_t)1.0 - dx) + src[y2 * width + x2] * dx;

        return val1 * ((channel_t)1.0 - dy) + val2 * dy;
    }
}

template<typename channel_t, typename color_t>
__global__ void importYUV(const void* devY, const void* devU,
                          const void* devV,
                          color_t* devYUV,
                          int inYw, int inYh,
                          int inUVw, int inUVh,
                          int outW, int outH,
                          float colorScale,
                          int cv_depth,
                          bool isSameYSize)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    channel_t yVal;
    if (isSameYSize)
    {
        uint32_t idx = y * inYw + x;
        if (cv_depth == CV_32F)
            yVal = static_cast<channel_t>(((float*)devY)[idx]);
        else if (cv_depth == CV_16U)
            yVal = static_cast<channel_t>(((unsigned short*)devY)[idx]);
        else
            yVal = static_cast<channel_t>(((unsigned char*)devY)[idx]);
    }
    else
    {
        float sx = float(x) * inYw / outW;
        float sy = float(y) * inYh / outH;
        uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
        uint16_t x2 = MIN(x1 + 1, inYw - 1);
        uint16_t y2 = MIN(y1 + 1, inYh - 1);
        float dx = sx - x1, dy = sy - y1;

        if (cv_depth == CV_32F)
            yVal = bilinearInterpolate<float, float>((float*)devY,
                inYw, inYh, x1, y1, x2, y2, dx, dy);
        else if (cv_depth == CV_16U)
            yVal = bilinearInterpolate<unsigned short, float>((unsigned short*)devY,
                inYw, inYh, x1, y1, x2, y2, dx, dy);
        else
            yVal = bilinearInterpolate<unsigned char, float>((unsigned char*)devY,
                inYw, inYh, x1, y1, x2, y2, dx, dy);
    }

    float scaleX = float(inUVw) / outW;
    float scaleY = float(inUVh) / outH;
    float sx = float(x) * scaleX;
    float sy = float(y) * scaleY;
    uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
    uint16_t x2 = MIN(x1 + 1, inUVw - 1);
    uint16_t y2 = MIN(y1 + 1, inUVh - 1);
    float dx = sx - x1, dy = sy - y1;

    channel_t uVal, vVal;
    if (cv_depth == CV_32F)
    {
        uVal = bilinearInterpolate<float, channel_t>((float*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vVal = bilinearInterpolate<float, channel_t>((float*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }
    else if (cv_depth == CV_16U)
    {
        uVal = bilinearInterpolate<unsigned short, channel_t>((unsigned short*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vVal = bilinearInterpolate<unsigned short, channel_t>((unsigned short*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }
    else
    {
        uVal = bilinearInterpolate<unsigned char, channel_t>((unsigned char*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vVal = bilinearInterpolate<unsigned char, channel_t>((unsigned char*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }

    uint32_t outIdx = y * outW + x;
    devYUV[outIdx].x = yVal * colorScale;
    devYUV[outIdx].y = uVal * colorScale;
    devYUV[outIdx].z = vVal * colorScale;
}

template<>
__global__ void importYUV<half, half3>(const void* devY, const void* devU,
                                       const void* devV, half3* devYUV,
                                       int inYw, int inYh,
                                       int inUVw, int inUVh,
                                       int outW, int outH,
                                       float colorScaleF,
                                       int cv_depth,
                                       bool isSameYSize)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    half hScale = __float2half(colorScaleF);

    half yVal;
    if (isSameYSize)
    {
        uint32_t idx = y * inYw + x;
        if (cv_depth == CV_32F)
            yVal = __float2half(((float*)devY)[idx]);
        else if (cv_depth == CV_16U)
            yVal = __float2half(static_cast<float>(((unsigned short*)devY)[idx]));
        else
            yVal = __float2half(static_cast<float>(((unsigned char*)devY)[idx]));
    }
    else
    {
        float sx = float(x) * inYw / outW;
        float sy = float(y) * inYh / outH;
        uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
        uint16_t x2 = MIN(x1 + 1, inYw - 1);
        uint16_t y2 = MIN(y1 + 1, inYh - 1);
        float dx = sx - x1, dy = sy - y1;

        if (cv_depth == CV_32F)
            yVal = bilinearInterpolate<float, half>((float*)devY, inYw, inYh, x1, y1, x2, y2, dx, dy);
        else if (cv_depth == CV_16U)
            yVal = bilinearInterpolate<unsigned short, half>((unsigned short*)devY, inYw, inYh, x1, y1, x2, y2, dx, dy);
        else
            yVal = bilinearInterpolate<unsigned char, half>((unsigned char*)devY, inYw, inYh, x1, y1, x2, y2, dx, dy);
    }

    float scaleX = float(inUVw) / outW;
    float scaleY = float(inUVh) / outH;
    float sx = float(x) * scaleX;
    float sy = float(y) * scaleY;
    uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
    uint16_t x2 = MIN(x1 + 1, inUVw - 1);
    uint16_t y2 = MIN(y1 + 1, inUVh - 1);
    float dx = sx - x1, dy = sy - y1;

    half uf, vf;
    if (cv_depth == CV_32F)
    {
        uf = bilinearInterpolate<float, half>((float*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vf = bilinearInterpolate<float, half>((float*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }
    else if (cv_depth == CV_16U)
    {
        uf = bilinearInterpolate<unsigned short, half>((unsigned short*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vf = bilinearInterpolate<unsigned short, half>((unsigned short*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }
    else
    {
        uf = bilinearInterpolate<unsigned char, half>((unsigned char*)devU, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
        vf = bilinearInterpolate<unsigned char, half>((unsigned char*)devV, inUVw, inUVh, x1, y1, x2, y2, dx, dy);
    }

    uint32_t outIdx = y * outW + x;
    devYUV[outIdx].Y = make_half2(yVal * hScale, __float2half(0.0f));
    devYUV[outIdx].UV = make_half2(uf * hScale, vf * hScale);
}

template<typename channel_t>
__global__ void importDepth(void* src,
                            channel_t* dst,
                            int inW, int inH,
                            int outW, int outH,
                            float scalePix,
                            float near, float far,
                            int cv_depth,
                            bool hasInvalidDepth,
                            bool isSameSize)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    channel_t depth;
    if (isSameSize)
    {
        uint32_t idx = y * inW + x;
        if (cv_depth == CV_32F)
            depth = static_cast<channel_t>(((float*)src)[idx]) * scalePix;
        else if (cv_depth == CV_16U)
            depth = static_cast<channel_t>(((unsigned short*)src)[idx]) * scalePix;
        else
            depth = static_cast<channel_t>(((unsigned char*)src)[idx]) * scalePix;
    }
    else
    {
        float sx = float(x) * inW / outW;
        float sy = float(y) * inH / outH;
        uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
        uint16_t x2 = MIN(x1 + 1, inW - 1);
        uint16_t y2 = MIN(y1 + 1, inH - 1);
        float dx = sx - x1, dy = sy - y1;

        if (cv_depth == CV_32F)
            depth = bilinearInterpolate<float, channel_t>((float*)src, inW, inH, x1, y1, x2, y2, dx, dy)* scalePix;
        else if (cv_depth == CV_16U)
            depth = bilinearInterpolate<unsigned short, channel_t>((unsigned short*)src, inW, inH, x1, y1, x2, y2, dx, dy)* scalePix;
        else
            depth = bilinearInterpolate<unsigned char, channel_t>((unsigned char*)src, inW, inH, x1, y1, x2, y2, dx, dy)* scalePix;
    }

    if (far >= 1000.f)
        depth = near / depth;
    else
        depth = far * near / (near + depth * (far - near));

    if (hasInvalidDepth && depth == 0.0f)
        depth = NAN;

    dst[y * outW + x] = depth;
}

template<>
__global__ void importDepth<half>(void* src, half* dst,
                                  int inW, int inH,
                                  int outW, int outH,
                                  float scalePixF,
                                  float nearF, float farF,
                                  int cv_depth,
                                  bool hasInvalidDepth,
                                  bool isSameSize)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    half SCALE = __float2half(scalePixF);
    half NEAR = __float2half(nearF);
    half FAR = __float2half(farF);
    half THRES = __float2half(1000.f);

    half depth;
    if (isSameSize)
    {
        uint32_t idx = y * inW + x;
        if (cv_depth == CV_32F)
            depth = __float2half(((float*)src)[idx]) * SCALE;
        else if (cv_depth == CV_16U)
            depth = __float2half(((unsigned short*)src)[idx]) * SCALE;
        else
            depth = __float2half(((unsigned char*)src)[idx]) * SCALE;
    }
    else
    {
        float sx = float(x) * inW / outW;
        float sy = float(y) * inH / outH;
        uint16_t x1 = (uint16_t)sx, y1 = (uint16_t)sy;
        uint16_t x2 = MIN(x1 + 1, inW - 1);
        uint16_t y2 = MIN(y1 + 1, inH - 1);
        float dx = sx - x1, dy = sy - y1;

        if (cv_depth == CV_32F)
            depth = bilinearInterpolate<float, half>((float*)src, inW, inH, x1, y1, x2, y2, dx, dy);
        else if (cv_depth == CV_16U)
            depth = bilinearInterpolate<unsigned short, half>((unsigned short*)src, inW, inH, x1, y1, x2, y2, dx, dy);
        else
            depth = bilinearInterpolate<unsigned char, half>((unsigned char*)src, inW, inH, x1, y1, x2, y2, dx, dy);

        depth = depth * SCALE;
    }

    if (FAR >= THRES)
        depth = NEAR / depth;
    else
        depth = FAR * NEAR / (NEAR + depth * (FAR - NEAR));

    if (hasInvalidDepth && depth == __float2half(0.f))
        depth = hNAN;

    dst[y * outW + x] = depth;
}

__device__ inline float bilinearInterpolateFromValues(float v11, float v21, float v12, float v22, float dx, float dy)
{
    float val1 = v11 * (1.0f - dx) + v21 * dx;
    float val2 = v12 * (1.0f - dx) + v22 * dx;

    return val1 * (1.0f - dy) + val2 * dy;
}

__device__ inline double bilinearInterpolateFromValues(double v11, double v21, double v12, double v22, float dx, float dy)
{
    double val1 = v11 * (1.0 - dx) + v21 * dx;
    double val2 = v12 * (1.0 - dx) + v22 * dx;

    return val1 * (1.0f - dy) + val2 * dy;
}

__device__ inline half2 bilinearInterpolateFromValues(half2 v11, half2 v21, half2 v12, half2 v22, float dx, float dy)
{
    half2 Hdx = __floats2half2_rn(dx, 1.0f - dx);
    half2 Hdy = __floats2half2_rn(dy, 1.0f - dy);

    half2 coeffX0 = make_half2(Hdx.y, Hdx.y);
    half2 coeffX1 = make_half2(Hdx.x, Hdx.x);
    half2 coeffY0 = make_half2(Hdy.y, Hdy.y);
    half2 coeffY1 = make_half2(Hdy.x, Hdy.x);

    half2 a = v11 * coeffX0 + v21 * coeffX1;
    half2 b = v12 * coeffX0 + v22 * coeffX1;

    return a * coeffY0 + b * coeffY1;
}

template<typename channel_t, typename color_t>
__global__ void exportYUV(const color_t* devYUV,
                          void* devDstY,
                          void* devDstU,
                          void* devDstV,
                          int inputWidth,
                          int inputHeight,
                          int outputYWidth,
                          int outputYHeight,
                          int outputUVWidth,
                          int outputUVHeight,
                          int cv_depth,
                          unsigned max_val,
                          bool isSameSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputUVWidth && y < outputUVHeight)
    {
        float scaleX = float(inputWidth) / outputUVWidth;
        float scaleY = float(inputHeight) / outputUVHeight;
        float srcX = x * scaleX;
        float srcY = y * scaleY;
        int x1 = int(srcX), y1 = int(srcY);
        int x2 = MIN(x1 + 1, inputWidth - 1);
        int y2 = MIN(y1 + 1, inputHeight - 1);
        float dx = srcX - x1;
        float dy = srcY - y1;

        int idx11 = y1 * inputWidth + x1;
        int idx21 = y1 * inputWidth + x2;
        int idx12 = y2 * inputWidth + x1;
        int idx22 = y2 * inputWidth + x2;

        channel_t u11 = devYUV[idx11].y;
        channel_t u21 = devYUV[idx21].y;
        channel_t u12 = devYUV[idx12].y;
        channel_t u22 = devYUV[idx22].y;

        channel_t v11 = devYUV[idx11].z;
        channel_t v21 = devYUV[idx21].z;
        channel_t v12 = devYUV[idx12].z;
        channel_t v22 = devYUV[idx22].z;

        channel_t interpU = bilinearInterpolateFromValues(u11, u21, u12, u22, dx, dy) * max_val;
        channel_t interpV = bilinearInterpolateFromValues(v11, v21, v12, v22, dx, dy) * max_val;

        int outIdx = y * outputUVWidth + x;
        if (cv_depth == CV_8U)
        {
            ((unsigned char*)devDstU)[outIdx] = static_cast<unsigned char>(interpU);
            ((unsigned char*)devDstV)[outIdx] = static_cast<unsigned char>(interpV);
        }
        else if (cv_depth == CV_16U)
        {
            ((unsigned short*)devDstU)[outIdx] = static_cast<unsigned short>(interpU);
            ((unsigned short*)devDstV)[outIdx] = static_cast<unsigned short>(interpV);
        }
        else if (cv_depth == CV_32F)
        {
            ((float*)devDstU)[outIdx] = static_cast<float>(interpU);
            ((float*)devDstV)[outIdx] = static_cast<float>(interpV);
        }
    }

    if (x < outputYWidth && y < outputYHeight)
    {
        int outIdx = y * outputYWidth + x;
        if (isSameSize)
        {
            double value = devYUV[outIdx].x * max_val;
            if (cv_depth == CV_8U)
                ((unsigned char*)devDstY)[outIdx] = static_cast<unsigned char>(value);
            else if (cv_depth == CV_16U)
                ((unsigned short*)devDstY)[outIdx] = static_cast<unsigned short>(value);
            else if (cv_depth == CV_32F)
                ((float*)devDstY)[outIdx] = static_cast<float>(value);
        }
        else
        {
            float scaleX = float(inputWidth) / outputYWidth;
            float scaleY = float(inputHeight) / outputYHeight;
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            int x1 = int(srcX), y1 = int(srcY);
            int x2 = MIN(x1 + 1, inputWidth - 1);
            int y2 = MIN(y1 + 1, inputHeight - 1);
            float dx = srcX - x1;
            float dy = srcY - y1;

            channel_t v11 = devYUV[y1 * inputWidth + x1].x;
            channel_t v21 = devYUV[y1 * inputWidth + x2].x;
            channel_t v12 = devYUV[y2 * inputWidth + x1].x;
            channel_t v22 = devYUV[y2 * inputWidth + x2].x;

            channel_t interp = bilinearInterpolateFromValues(v11, v21, v12, v22, dx, dy) * max_val;

            if (cv_depth == CV_8U)
                ((unsigned char*)devDstY)[outIdx] = static_cast<unsigned char>(interp);
            else if (cv_depth == CV_16U)
                ((unsigned short*)devDstY)[outIdx] = static_cast<unsigned short>(interp);
            else if (cv_depth == CV_32F)
                ((float*)devDstY)[outIdx] = static_cast<float>(interp);
        }
    }
}

template<>
__global__ void exportYUV<half, half3>(const half3* devYUV,
                                       void* devDstY,
                                       void* devDstU,
                                       void* devDstV,
                                       int inputWidth,
                                       int inputHeight,
                                       int outputYWidth,
                                       int outputYHeight,
                                       int outputUVWidth,
                                       int outputUVHeight,
                                       int cv_depth,
                                       unsigned max_val,
                                       bool isSameSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    half hmax = __float2half((float)max_val);
    half2 hmax2 = make_half2(hmax, hmax);

    if (x < outputUVWidth && y < outputUVHeight)
    {
        float scaleX = float(inputWidth) / outputUVWidth;
        float scaleY = float(inputHeight) / outputUVHeight;
        float srcX = x * scaleX;
        float srcY = y * scaleY;
        int x1 = int(srcX), y1 = int(srcY);
        int x2 = MIN(x1 + 1, inputWidth - 1);
        int y2 = MIN(y1 + 1, inputHeight - 1);
        float dx = srcX - x1;
        float dy = srcY - y1;

        int idx11 = y1 * inputWidth + x1;
        int idx21 = y1 * inputWidth + x2;
        int idx12 = y2 * inputWidth + x1;
        int idx22 = y2 * inputWidth + x2;

        half2 uv11 = devYUV[idx11].UV;
        half2 uv21 = devYUV[idx21].UV;
        half2 uv12 = devYUV[idx12].UV;
        half2 uv22 = devYUV[idx22].UV;

        half2 interpUV = bilinearInterpolateFromValues(uv11, uv21, uv12, uv22, dx, dy) * hmax2;

        half quantU = interpUV.x;
        half quantV = interpUV.y;

        int outIdx = y * outputUVWidth + x;
        if (cv_depth == CV_8U)
        {
            ((unsigned char*)devDstU)[outIdx] = static_cast<unsigned char>(__half2float(quantU));
            ((unsigned char*)devDstV)[outIdx] = static_cast<unsigned char>(__half2float(quantV));
        }
        else if (cv_depth == CV_16U)
        {
            ((unsigned short*)devDstU)[outIdx] = static_cast<unsigned short>(__half2float(quantU));
            ((unsigned short*)devDstV)[outIdx] = static_cast<unsigned short>(__half2float(quantV));
        }
        else if (cv_depth == CV_32F)
        {
            ((float*)devDstU)[outIdx] = __half2float(quantU);
            ((float*)devDstV)[outIdx] = __half2float(quantV);
        }
    }

    if (x < outputYWidth && y < outputYHeight)
    {
        int outIdx = y * outputYWidth + x;
        if (isSameSize)
        {
            half2 yVal2 = devYUV[outIdx].Y;
            half quantY = yVal2.x * hmax;
            if (cv_depth == CV_8U)
                ((unsigned char*)devDstY)[outIdx] = static_cast<unsigned char>(__half2float(quantY));
            else if (cv_depth == CV_16U)
                ((unsigned short*)devDstY)[outIdx] = static_cast<unsigned short>(__half2float(quantY));
            else if (cv_depth == CV_32F)
                ((float*)devDstY)[outIdx] = __half2float(quantY);
        }
        else
        {
            float scaleX = float(inputWidth) / outputYWidth;
            float scaleY = float(inputHeight) / outputYHeight;
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            int x1 = int(srcX), y1 = int(srcY);
            int x2 = MIN(x1 + 1, inputWidth - 1);
            int y2 = MIN(y1 + 1, inputHeight - 1);
            float dx = srcX - x1;
            float dy = srcY - y1;

            half2 y11 = devYUV[y1 * inputWidth + x1].Y;
            half2 y21 = devYUV[y1 * inputWidth + x2].Y;
            half2 y12 = devYUV[y2 * inputWidth + x1].Y;
            half2 y22 = devYUV[y2 * inputWidth + x2].Y;

            half interpY = bilinearInterpolateFromValues(y11, y21, y12, y22, dx, dy).x * hmax;

            if (cv_depth == CV_8U)
                ((unsigned char*)devDstY)[outIdx] = static_cast<unsigned char>(__half2float(interpY));
            else if (cv_depth == CV_16U)
                ((unsigned short*)devDstY)[outIdx] = static_cast<unsigned short>(__half2float(interpY));
            else if (cv_depth == CV_32F)
                ((float*)devDstY)[outIdx] = __half2float(interpY);
        }
    }
}

template<typename channel_t, typename color_t>
void importColorsToGPU(cv::Mat& hostYUV, color_t*& devNormalizedYUV,
                       cv::Size initialY_size, cv::Size realSize,
                       int type, float colorScale,
                       cudaStream_t& stream, size_t yInputBytes, size_t uvInputBytes,
                       void*& devReadYUV, cudaEvent_t& importColorFinished)
{
    cudaMemcpyAsync(devReadYUV, hostYUV.data, yInputBytes + 2 * uvInputBytes, cudaMemcpyHostToDevice, stream);

    int bw = 16, bh = 8;
    dim3 block(bw, bh);
    dim3 grid((realSize.width + bw - 1) / bw,
        (realSize.height + bh - 1) / bh);

    cv::Size uvSize = initialY_size / 2;

    const uint8_t* base = static_cast<uint8_t*>(devReadYUV);
    const void* devY = base;
    const void* devU = base + yInputBytes;
    const void* devV = base + yInputBytes + uvInputBytes;

    importYUV<channel_t, color_t><<<grid, block, 0, stream>>>(
        devY, devU, devV,
        devNormalizedYUV,
        initialY_size.width, initialY_size.height,
        uvSize.width, uvSize.height,
        realSize.width, realSize.height,
        colorScale, type, (initialY_size == realSize));

    cudaEventRecord(importColorFinished, stream);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template<typename channel_t>
void importDepthToGPU(cv::Mat& hostDepth, channel_t*& devNormalizedDepth, cv::Size initialDepthSize, cv::Size realSize,
                      float scale, float near, float far, bool hasInvalidDepth, int type,
                      cudaStream_t& stream,
                      size_t inputDepthBytes,
                      void*& devReadDepth,
                      cudaEvent_t& importDepthFinished)
{
    cudaMemcpyAsync(devReadDepth, hostDepth.data, inputDepthBytes, cudaMemcpyHostToDevice, stream);

    int bw = 16, bh = 8;
    dim3 block(bw, bh);
    dim3 grid((realSize.width + bw - 1) / bw,
        (realSize.height + bh - 1) / bh);

    importDepth<channel_t><<<grid, block, 0, stream>>>(devReadDepth, devNormalizedDepth,
                                                       hostDepth.cols, hostDepth.rows,
                                                       realSize.width, realSize.height,
                                                       scale, near, far, type,
                                                       hasInvalidDepth,
                                                       (initialDepthSize == realSize));

    cudaEventRecord(importDepthFinished, stream);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template<typename channel_t, typename color_t>
void exportColorsToCPU(color_t*& devYUV, cv::Mat& hostY, cv::Mat& hostU, cv::Mat& hostV,
                       cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
                       int cv_depth, unsigned max_val,
                       cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
                       cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
                       size_t dstSizeY, size_t dstSizeUV,
                       void*& devDstY, void*& devDstU, void*& devDstV)
{
    dim3 blockDim(16, 8);
    int maxW = MAX(outputUV_size.width, outputY_size.width);
    int maxH = MAX(outputUV_size.height, outputY_size.height);
    dim3 gridDim((maxW + blockDim.x - 1) / blockDim.x,
                 (maxH + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, inpainted, 0);

    exportYUV<channel_t, color_t><<<gridDim, blockDim, 0, stream1>>>(devYUV,
                                                                     devDstY, devDstU, devDstV,
                                                                     virtualSize.width, virtualSize.height,
                                                                     outputY_size.width, outputY_size.height,
                                                                     outputUV_size.width, outputUV_size.height,
                                                                     cv_depth, max_val,
                                                                     (virtualSize == outputY_size));
    cudaEventRecord(exportedUV, stream1);

    cudaMemcpyAsync(hostU.data, devDstU, dstSizeUV, cudaMemcpyDeviceToHost, stream1);

    cudaStreamWaitEvent(stream2, exportedUV, 0);
    cudaMemcpyAsync(hostV.data, devDstV, dstSizeUV, cudaMemcpyDeviceToHost, stream2);

    cudaStreamWaitEvent(stream3, exportedUV, 0);
    cudaMemcpyAsync(hostY.data, devDstY, dstSizeY, cudaMemcpyDeviceToHost, stream3);

    cudaEventRecord(writeU, stream1);
    cudaEventRecord(writeV, stream2);
    cudaEventRecord(writeY, stream3);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template void importColorsToGPU<float, float3>(
    cv::Mat& hostYUV, float3*& devNormalizedYUV,
    cv::Size initialY_size, cv::Size realSize,
    int type, float colorScale,
    cudaStream_t& stream, size_t yuvInputBytes, size_t uvInputBytes,
    void*& devReadYUV, cudaEvent_t& importColorFinished);

template void importColorsToGPU<double, double3>(
    cv::Mat& hostYUV, double3*& devNormalizedYUV,
    cv::Size initialY_size, cv::Size realSize,
    int type, float colorScale,
    cudaStream_t& stream, size_t yuvInputBytes, size_t uvInputBytes,
    void*& devReadYUV, cudaEvent_t& importColorFinished);

template void importColorsToGPU<half, half3>(
    cv::Mat& hostYUV, half3*& devNormalizedYUV,
    cv::Size initialY_size, cv::Size realSize,
    int type, float colorScale,
    cudaStream_t& stream, size_t yuvInputBytes, size_t uvInputBytes,
    void*& devReadYUV, cudaEvent_t& importColorFinished);

template void importDepthToGPU<float>(cv::Mat& hostDepth, float*& devNormalizedDepth, cv::Size initialDepthSize, cv::Size realSize,
                                      float scale, float near, float far, bool hasInvalidDepth, int type,
                                      cudaStream_t& stream,
                                      size_t inputDepthBytes,
                                      void*& devReadDepth,
                                      cudaEvent_t& importDepthFinished);

template void importDepthToGPU<double>(cv::Mat& hostDepth, double*& devNormalizedDepth, cv::Size initialDepthSize, cv::Size realSize,
                                      float scale, float near, float far, bool hasInvalidDepth, int type,
                                      cudaStream_t& stream,
                                      size_t inputDepthBytes,
                                      void*& devReadDepth,
                                      cudaEvent_t& importDepthFinished);

template void importDepthToGPU<half>(cv::Mat& hostDepth, half*& devNormalizedDepth, cv::Size initialDepthSize, cv::Size realSize,
                                     float scale, float near, float far, bool hasInvalidDepth, int type,
                                     cudaStream_t& stream,
                                     size_t inputDepthBytes,
                                     void*& devReadDepth,
                                     cudaEvent_t& importDepthFinished);

template void exportColorsToCPU<float, float3>(float3*& devYUV, cv::Mat& hostY, cv::Mat& hostU, cv::Mat& hostV,
    cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
    int cv_depth, unsigned max_val,
    cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
    cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
    size_t dstSizeY, size_t dstSizeUV,
    void*& devDstY, void*& devDstU, void*& devDstV);

template void exportColorsToCPU<double, double3>(double3*& devYUV, cv::Mat& hostY, cv::Mat& hostU, cv::Mat& hostV,
    cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
    int cv_depth, unsigned max_val,
    cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
    cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
    size_t dstSizeY, size_t dstSizeUV,
    void*& devDstY, void*& devDstU, void*& devDstV);

template void exportColorsToCPU<half, half3>(half3*& devYUV, cv::Mat& hostY, cv::Mat& hostU, cv::Mat& hostV,
    cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
    int cv_depth, unsigned max_val,
    cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
    cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
    size_t dstSizeY, size_t dstSizeUV,
    void*& devDstY, void*& devDstU, void*& devDstV);