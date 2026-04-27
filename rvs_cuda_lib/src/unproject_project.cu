/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
    Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "unproject_project.cuh"


template<typename position_t, typename channel_t>
__global__ void unprojectERP_projectERP_kernel(channel_t* depth, channel_t* virtualDepth, position_t* imagePos,
                                               float phi0, float theta0, float dphi_du, float dtheta_dv,
                                               float u0, float v0, float du_dphi, float dv_dtheta,
                                               CamData camData, int imgWidth, int imgHeight)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    vec2_t<channel_t> uv;
    if (ty == 0)
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)0 };
    else if (ty == imgHeight - 1)
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)imgHeight };
    else
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)ty + (channel_t)0.5 };

    channel_t phi = phi0 + dphi_du * uv.x;
    channel_t theta = theta0 + dtheta_dv * uv.y;
    channel_t depthValue = depth[threadId];

    if (!isnan(depthValue))
    {
        vec3_t<channel_t> sphericalPos =
        {
            depthValue * cos(theta) * cos(phi),
            depthValue * cos(theta) * sin(phi),
            depthValue * sin(theta)
        };

        vec3_t<channel_t> xyz =
        {
            camData.rotation[0] * sphericalPos.x + camData.rotation[1] * sphericalPos.y + camData.rotation[2] * sphericalPos.z + camData.translation[0],
            camData.rotation[3] * sphericalPos.x + camData.rotation[4] * sphericalPos.y + camData.rotation[5] * sphericalPos.z + camData.translation[1],
            camData.rotation[6] * sphericalPos.x + camData.rotation[7] * sphericalPos.y + camData.rotation[8] * sphericalPos.z + camData.translation[2]
        };

        channel_t radius = xyz.x * xyz.x + xyz.y * xyz.y + xyz.z * xyz.z;

        if (radius > (channel_t)1e-6)
        {
            radius = sqrtf(radius);
            phi = atan2f(xyz.y, xyz.x);
            theta = asinf(xyz.z / radius);

            channel_t posx = u0 + du_dphi * phi;
            channel_t posy = v0 + dv_dtheta * theta;

            if (posx >= (channel_t)1e-6 && posy >= (channel_t)1e-6)
            {
                virtualDepth[threadId] = radius;
                imagePos[threadId] = { posx, posy };
                return;
            }
        }
    }

    virtualDepth[threadId] = (channel_t)NAN;
    imagePos[threadId] = { (channel_t)NAN, (channel_t)NAN };
}

template<>
__global__ void unprojectERP_projectERP_kernel<half2, half>(half* depth, half* virtualDepth, half2* imagePos,
                                                            float phi0, float theta0, float dphi_du, float dtheta_dv,
                                                            float u0, float v0, float du_dphi, float dv_dtheta,
                                                            CamData camData, int imgWidth, int imgHeight)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;
    half2 uv;
    if (ty == 0)
        uv = __floats2half2_rn((float)tx + 0.5f, 0.0f);
    else if (ty == imgHeight - 1)
        uv = __floats2half2_rn((float)tx + 0.5f, (float)imgHeight);
    else
        uv = __floats2half2_rn((float)tx + 0.5f, (float)ty + 0.5f);

    half phi = __float2half(phi0) + __float2half(dphi_du) * uv.x;
    half theta = __float2half(theta0) + __float2half(dtheta_dv) * uv.y;
    half depthValue = depth[threadId];

    if (!__hisnan(depthValue))
    {
        half3 sphericalPos;
        sphericalPos.Y.x = depthValue * hcos(theta) * hcos(phi);
        sphericalPos.UV.x = depthValue * hcos(theta) * hsin(phi);
        sphericalPos.UV.y = depthValue * hsin(theta);

        half3 xyz;
        xyz.Y.x = __float2half(camData.rotation[0]) * sphericalPos.Y.x +
                  __float2half(camData.rotation[1]) * sphericalPos.UV.x +
                  __float2half(camData.rotation[2]) * sphericalPos.UV.y +
                  __float2half(camData.translation[0]);

        xyz.UV.x = __float2half(camData.rotation[3]) * sphericalPos.Y.x +
                   __float2half(camData.rotation[4]) * sphericalPos.UV.x +
                   __float2half(camData.rotation[5]) * sphericalPos.UV.y +
                   __float2half(camData.translation[1]);

        xyz.UV.y = __float2half(camData.rotation[6]) * sphericalPos.Y.x +
                   __float2half(camData.rotation[7]) * sphericalPos.UV.x +
                   __float2half(camData.rotation[8]) * sphericalPos.UV.y +
                   __float2half(camData.translation[2]);

        half radius = xyz.Y.x * xyz.Y.x + xyz.UV.x * xyz.UV.x + xyz.UV.y * xyz.UV.y;

        half EPS = __float2half(5e-6f);
        if (radius > EPS)
        {
            radius = hsqrt(radius);

            phi = __float2half(atan2f(__half2float(xyz.UV.x), __half2float(xyz.Y.x)));
            theta = __float2half(asinf(__half2float(xyz.UV.y) / __half2float(radius)));

            half posx = __float2half(u0) + __float2half(du_dphi) * phi;
            half posy = __float2half(v0) + __float2half(dv_dtheta) * theta;

            if (posx >= EPS && posy >= EPS)
            {
                virtualDepth[threadId] = radius;
                imagePos[threadId] = make_half2(posx, posy);
                return;
            }
        }
    }

    virtualDepth[threadId] = hNAN;
    imagePos[threadId] = make_half2(hNAN, hNAN);
}

template<typename position_t, typename channel_t>
__global__ void unprojectERP_projectPerspective_kernel(channel_t* depth, channel_t* virtualDepth, position_t* imagePos,
                                                       float phi0, float theta0, float dphi_du, float dtheta_dv,
                                                       CamData camData, int imgWidth, int imgHeight)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;

    vec2_t<channel_t> uv;
    if (ty == 0)
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)0 };
    else if (ty == imgHeight - 1)
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)imgHeight };
    else
        uv = { (channel_t)tx + (channel_t)0.5, (channel_t)ty + (channel_t)0.5 };

    channel_t phi = phi0 + dphi_du * uv.x;
    channel_t theta = theta0 + dtheta_dv * uv.y;
    channel_t depthValue = depth[threadId];

    if (!isNaN(depthValue))
    {
        vec3_t<channel_t> sphericalPos =
        {
            depthValue * cos(theta) * cos(phi),
            depthValue * cos(theta) * sin(phi),
            depthValue * sin(theta)
        };

        vec3_t<channel_t> xyz =
        {
            camData.rotation[0] * sphericalPos.x + camData.rotation[1] * sphericalPos.y + camData.rotation[2] * sphericalPos.z + camData.translation[0],
            camData.rotation[3] * sphericalPos.x + camData.rotation[4] * sphericalPos.y + camData.rotation[5] * sphericalPos.z + camData.translation[1],
            camData.rotation[6] * sphericalPos.x + camData.rotation[7] * sphericalPos.y + camData.rotation[8] * sphericalPos.z + camData.translation[2]
        };

        if (xyz.x > (channel_t)1e-6)
        {
            channel_t posx = -camData.focal[0] * xyz.y / xyz.x + camData.principlePoint[0];
            channel_t posy = -camData.focal[1] * xyz.z / xyz.x + camData.principlePoint[1];

            if (posx >= (channel_t)1e-6 && posy >= (channel_t)1e-6)
            {
                imagePos[threadId] = { posx, posy };
                virtualDepth[threadId] = xyz.x;
                return;
            }
        }
    }

    virtualDepth[threadId] = NAN;
    imagePos[threadId] = { NAN, NAN };
}

template<>
__global__ void unprojectERP_projectPerspective_kernel<half2, half>(half* depth, half* virtualDepth, half2* imagePos,
                                                       float phi0, float theta0, float dphi_du, float dtheta_dv,
                                                       CamData camData, int imgWidth, int imgHeight)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgWidth || ty >= imgHeight)
        return;

    int threadId = ty * imgWidth + tx;
    half2 uv;
    if (ty == 0)
        uv = __floats2half2_rn((float)tx + 0.5f, 0.0f);
    else if (ty == imgHeight - 1)
        uv = __floats2half2_rn((float)tx + 0.5f, (float)imgHeight);
    else
        uv = __floats2half2_rn((float)tx + 0.5f, (float)ty + 0.5f);

    half phi = __float2half(phi0) + __float2half(dphi_du) * uv.x;
    half theta = __float2half(theta0) + __float2half(dtheta_dv) * uv.y;
    half depthValue = depth[threadId];

    if (!isNaN(depthValue))
    {
        half3 sphericalPos;
        sphericalPos.Y.x = depthValue * hcos(theta) * hcos(phi);
        sphericalPos.UV.x = depthValue * hcos(theta) * hsin(phi);
        sphericalPos.UV.y = depthValue * hsin(theta);

        half3 xyz;
        xyz.Y.x = __float2half(camData.rotation[0]) * sphericalPos.Y.x +
                  __float2half(camData.rotation[1]) * sphericalPos.UV.x +
                  __float2half(camData.rotation[2]) * sphericalPos.UV.y +
                  __float2half(camData.translation[0]);

        xyz.UV.x = __float2half(camData.rotation[3]) * sphericalPos.Y.x +
                   __float2half(camData.rotation[4]) * sphericalPos.UV.x +
                   __float2half(camData.rotation[5]) * sphericalPos.UV.y +
                   __float2half(camData.translation[1]);

        xyz.UV.y = __float2half(camData.rotation[6]) * sphericalPos.Y.x +
                   __float2half(camData.rotation[7]) * sphericalPos.UV.x +
                   __float2half(camData.rotation[8]) * sphericalPos.UV.y +
                   __float2half(camData.translation[2]);

        half EPS = __float2half(6e-5f);
        if (xyz.Y.x > EPS)
        {
            half posx = __float2half(-camData.focal[0]) * xyz.UV.x / xyz.Y.x +
                        __float2half(camData.principlePoint[0]);

            half posy = __float2half(-camData.focal[1]) * xyz.UV.y / xyz.Y.x +
                        __float2half(camData.principlePoint[1]);

            if (posx >= EPS && posy >= EPS)
            {
                imagePos[threadId] = make_half2(posx, posy);
                virtualDepth[threadId] = xyz.Y.x;
                return;
            }
        }
    }

    virtualDepth[threadId] = hNAN;
    imagePos[threadId] = make_half2(hNAN, hNAN);
}

template<typename position_t, typename channel_t>
void unprojectERP_projectERP(cv::Size size,
                             channel_t*& devDepth, position_t*& devTransformedPosition, channel_t*& devTransformedDepth,
                             const PrecomputedParams& params,
                             cudaStream_t& stream)
{
    unprojectERP_projectERP_kernel<position_t, channel_t><<<params.gridDim, params.blockDim, 0, stream>>>(
        devDepth, devTransformedDepth, devTransformedPosition,
        params.devPhi0, params.devTheta0,
        params.dev_dphi_du, params.dev_dtheta_dv,
        params.devU0, params.devV0,
        params.dev_du_dphi, params.dev_dv_dtheta,
        params.camData,
        size.width, size.height);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template<typename position_t, typename channel_t>
void unprojectERP_projectPerspective(cv::Size size,
                                     channel_t*& devDepth, position_t*& devTransformedPosition, channel_t*& devTransformedDepth,
                                     const PrecomputedParams& params,
                                     cudaStream_t& stream)
{
    unprojectERP_projectPerspective_kernel<position_t, channel_t><<<params.gridDim, params.blockDim, 0, stream>>>(
        devDepth, devTransformedDepth, devTransformedPosition,
        params.devPhi0, params.devTheta0,
        params.dev_dphi_du, params.dev_dtheta_dv,
        params.camData,
        size.width, size.height);

#ifdef _DEBUG
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

template void unprojectERP_projectERP<float2, float>(cv::Size size,
    float*& devDepth, float2*& devTransformedPosition, float*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);

template void unprojectERP_projectERP<double2, double>(cv::Size size,
    double*& devDepth, double2*& devTransformedPosition, double*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);

template void unprojectERP_projectERP<half2, half>(cv::Size size,
    half*& devDepth, half2*& devTransformedPosition, half*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);

template void unprojectERP_projectPerspective<float2, float>(cv::Size size,
    float*& devDepth, float2*& devTransformedPosition, float*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);

template void unprojectERP_projectPerspective<double2, double>(cv::Size size,
    double*& devDepth, double2*& devTransformedPosition, double*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);

template void unprojectERP_projectPerspective<half2, half>(cv::Size size,
    half*& devDepth, half2*& devTransformedPosition, half*& devTransformedDepth,
    const PrecomputedParams& params,
    cudaStream_t& stream);
