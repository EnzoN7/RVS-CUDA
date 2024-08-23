#include "unproject_project.cuh"

/**
 * @brief A constant structure to hold the dimensions of the image (or the grid) being processed.
 */
__constant__ int imgDim[2];

/**
 * @brief A constant 3x3 rotation matrix.
 */
__constant__ float rotation[9];

/**
 * @brief A constant 3D translation vector.
 */
__constant__ float translation[3];

/**
 * @brief A constant array holding the focal lengths for the x and y axes.
 */
__constant__ float focal[2];

/**
 * @brief A constant array holding the principal points for the x and y axes.
 */
__constant__ float principlePoint[2];


/**
 * @brief CUDA kernel for unprojecting ERP coordinates to 3D space and reprojecting to ERP coordinates.
 *
 * @param depth Input depth image as a single-channel float array.
 * @param virtualDepth Output virtual depth image as a single-channel float array.
 * @param imagePos Output image positions as a 2D float array.
 * @param phi0 Initial phi angle in radians.
 * @param theta0 Initial theta angle in radians.
 * @param dphi_du Change in phi per unit u.
 * @param dtheta_dv Change in theta per unit v.
 * @param u0 Initial u coordinate.
 * @param v0 Initial v coordinate.
 * @param du_dphi Change in u per unit phi.
 * @param dv_dtheta Change in v per unit theta.
 */
__global__ void unprojectERP_projectERP_kernel(float* depth, float* virtualDepth, float2* imagePos, float phi0, float theta0, float dphi_du, float dtheta_dv, float u0, float v0, float du_dphi, float dv_dtheta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgDim[0] && ty >= imgDim[1])
        return;

    int threadId = ty * imgDim[0] + tx;

    float2 uv;
    if (ty == 0)
        uv = make_float2(tx + 0.5f, 0);
    else if (ty == imgDim[1] - 1)
        uv = make_float2(tx + 0.5f, imgDim[1]);
    else
        uv = make_float2(tx + 0.5f, ty + 0.5f);

    float phi = phi0 + dphi_du * uv.x;
    float theta = theta0 + dtheta_dv * uv.y;
    float depthValue = depth[threadId];

    float3 sphericalPos = make_float3
    (
        depthValue * cos(theta) * cos(phi),
        depthValue * cos(theta) * sin(phi),
        depthValue * sin(theta)
    );

    float3 xyz = make_float3
    (
        rotation[0] * sphericalPos.x + rotation[1] * sphericalPos.y + rotation[2] * sphericalPos.z + translation[0],
        rotation[3] * sphericalPos.x + rotation[4] * sphericalPos.y + rotation[5] * sphericalPos.z + translation[1],
        rotation[6] * sphericalPos.x + rotation[7] * sphericalPos.y + rotation[8] * sphericalPos.z + translation[2]
    );

    float radius = sqrtf(xyz.x * xyz.x + xyz.y * xyz.y + xyz.z * xyz.z);
    virtualDepth[threadId] = radius;

    phi = atan2f(xyz.y, xyz.x);
    theta = asinf(xyz.z / radius);

    imagePos[threadId] = make_float2(u0 + du_dphi * phi, v0 + dv_dtheta * theta);
}

/**
 * @brief CUDA kernel for unprojecting ERP coordinates to 3D space and reprojecting to perspective coordinates.
 *
 * @param depth Input depth image as a single-channel float array.
 * @param virtualDepth Output virtual depth image as a single-channel float array.
 * @param imagePos Output image positions as a 2D float array.
 * @param phi0 Initial phi angle in radians.
 * @param theta0 Initial theta angle in radians.
 * @param dphi_du Change in phi per unit u.
 * @param dtheta_dv Change in theta per unit v.
 */
__global__ void unprojectERP_projectPerspective_kernel(float* depth, float* virtualDepth, float2* imagePos, float phi0, float theta0, float dphi_du, float dtheta_dv)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= imgDim[0] && ty >= imgDim[1])
        return;

    int threadId = ty * imgDim[0] + tx;

    float2 uv;
    if (ty == 0)
        uv = make_float2(tx + 0.5f, 0);
    else if (ty == imgDim[1] - 1)
        uv = make_float2(tx + 0.5f, imgDim[1]);
    else
        uv = make_float2(tx + 0.5f, ty + 0.5f);

    float phi = phi0 + dphi_du * uv.x;
    float theta = theta0 + dtheta_dv * uv.y;
    float depthValue = depth[threadId];

    float3 sphericalPos = make_float3
    (
        depthValue * cos(theta) * cos(phi),
        depthValue * cos(theta) * sin(phi),
        depthValue * sin(theta)
    );

    float3 xyz = make_float3
    (
        rotation[0] * sphericalPos.x + rotation[1] * sphericalPos.y + rotation[2] * sphericalPos.z + translation[0],
        rotation[3] * sphericalPos.x + rotation[4] * sphericalPos.y + rotation[5] * sphericalPos.z + translation[1],
        rotation[6] * sphericalPos.x + rotation[7] * sphericalPos.y + rotation[8] * sphericalPos.z + translation[2]
    );

    if (xyz.x <= 0)
    {
        imagePos[threadId] = make_float2(NAN, NAN);
        virtualDepth[threadId] = NAN;
        return;
    }

    imagePos[threadId] = make_float2(-focal[0] * xyz.y / xyz.x + principlePoint[0], -focal[1] * xyz.z / xyz.x + principlePoint[1]);
    virtualDepth[threadId] = xyz.x;
}

/**
 * @brief Unprojects ERP coordinates to 3D space and reprojects to ERP coordinates.
 *
 * @param size The size of the input depth image.
 * @param depth Input depth image as a single-channel float matrix.
 * @param virtual_uv Output virtual UV coordinates as a reference to a 2-channel float matrix.
 * @param virtual_depth Output virtual depth as a reference to a single-channel float matrix.
 * @param hor_range The horizontal range in degrees.
 * @param ver_range The vertical range in degrees.
 * @param R The rotation matrix.
 * @param t The translation vector.
 */
void unprojectERP_projectERP
(
    cv::Size size, float* devDepth, float2*& devVirtualUV, float*& devVirtualDepth, cv::Vec2f hor_range, cv::Vec2f ver_range, cv::Matx33f R, cv::Vec3f t
)
{
    int imgHeight = size.height;
    int imgWidth = size.width;
    int imgSize = imgHeight * imgWidth;

    int hostImgDim[2] = { imgWidth, imgHeight };
    float hostRotation[9] = {
    R(0, 0), R(0, 1), R(0, 2),
    R(1, 0), R(1, 1), R(1, 2),
    R(2, 0), R(2, 1), R(2, 2)
    };
    float hostTranslation[3] = { t[0], t[1], t[2] };

    float radperdeg = 0.01745329252f;
    float devPhi0 = radperdeg * hor_range[1];
    float devTheta0 = radperdeg * ver_range[1];
    float dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;
    float dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

    const float degperrad = 57.295779513f;
    float devU0 = size.width * hor_range[1] / (hor_range[1] - hor_range[0]);
    float devV0 = size.height * ver_range[1] / (ver_range[1] - ver_range[0]);
    float dev_du_dphi = -degperrad * size.width / (hor_range[1] - hor_range[0]);
    float dev_dv_dtheta = -degperrad * size.height / (ver_range[1] - ver_range[0]);

    cudaMalloc(&devVirtualUV, imgSize * sizeof(float2));
    cudaMalloc(&devVirtualDepth, imgSize * sizeof(float));


    cudaMemcpyToSymbol(imgDim, &hostImgDim, 2 * sizeof(int));
    cudaMemcpyToSymbol(rotation, &hostRotation, 9 * sizeof(float));
    cudaMemcpyToSymbol(translation, &hostTranslation, 3 * sizeof(float));

    int blockWidth = 8;
    int blockHeight = 16;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    unprojectERP_projectERP_kernel <<<gridDim, blockDim>>> (devDepth, devVirtualDepth, devVirtualUV, devPhi0, devTheta0, dev_dphi_du, dev_dtheta_dv, devU0, devV0, dev_du_dphi, dev_dv_dtheta);

    cudaDeviceSynchronize();

    cudaFree(devDepth);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}

/**
 * @brief Unprojects ERP coordinates to 3D space and reprojects to perspective coordinates.
 *
 * @param size The size of the input depth image.
 * @param depth Input depth image as a single-channel float matrix.
 * @param virtual_uv Output virtual UV coordinates as a reference to a 2-channel float matrix.
 * @param virtual_depth Output virtual depth as a reference to a single-channel float matrix.
 * @param hor_range The horizontal range in degrees.
 * @param ver_range The vertical range in degrees.
 * @param R The rotation matrix.
 * @param t The translation vector.
 * @param f The focal lengths for the x and y axes.
 * @param p The principal points for the x and y axes.
 */
void unprojectERP_projectPerspective
(
    cv::Size size, float* devDepth, float2*& devVirtualUV, float*& devVirtualDepth, cv::Vec2f hor_range, cv::Vec2f ver_range, cv::Matx33f R, cv::Vec3f t, cv::Vec2f f, cv::Vec2f p
)
{
    int imgHeight = size.height;
    int imgWidth = size.width;
    int imgSize = imgHeight * imgWidth;

    int hostImgDim[2] = { imgWidth, imgHeight };
    float hostFocal[2] = { f[0], f[1] };
    float hostPrinciplePoint[2] = { p[0], p[1] };
    float hostRotation[9] = {
    R(0, 0), R(0, 1), R(0, 2),
    R(1, 0), R(1, 1), R(1, 2),
    R(2, 0), R(2, 1), R(2, 2)
    };
    float hostTranslation[3] = { t[0], t[1], t[2] };

    float radperdeg = 0.01745329252f;
    float devPhi0 = radperdeg * hor_range[1];
    float devTheta0 = radperdeg * ver_range[1];
    float dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;
    float dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

    cudaMalloc(&devVirtualUV, imgSize * sizeof(float2));
    cudaMalloc(&devVirtualDepth, imgSize * sizeof(float));

    cudaMemcpyToSymbol(imgDim, &hostImgDim, 2 * sizeof(int));
    cudaMemcpyToSymbol(focal, &hostFocal, 2 * sizeof(float));
    cudaMemcpyToSymbol(principlePoint, &hostPrinciplePoint, 2 * sizeof(float));
    cudaMemcpyToSymbol(rotation, &hostRotation, 9 * sizeof(float));
    cudaMemcpyToSymbol(translation, &hostTranslation, 3 * sizeof(float));

    int blockWidth = 8;
    int blockHeight = 16;
    dim3 gridDim((imgWidth - 1 + blockWidth) / blockWidth, (imgHeight - 1 + blockHeight) / blockHeight);
    dim3 blockDim(blockWidth, blockHeight);

    unprojectERP_projectPerspective_kernel <<<gridDim, blockDim>>> (devDepth, devVirtualDepth, devVirtualUV, devPhi0, devTheta0, dev_dphi_du, dev_dtheta_dv);

    cudaDeviceSynchronize();

    cudaFree(devDepth);

    cudaError_t state = cudaGetLastError();
    bool error = state != cudaSuccess;
    if (error)
        throw std::runtime_error(cudaGetErrorString(state));
}
