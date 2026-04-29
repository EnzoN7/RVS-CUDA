#ifndef BLEND_IMG_CUH
#define BLEND_IMG_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Auteur : Enzo DI MARIA
*/

void blendImages
(
    float3* devBlendedColor, float* devBlendedValidity, float* devBlendedDepth,
    float3* devColor, float* devValidity, float* devDepth, cv::Size outputSize,
    cv::Vec3f emptyColor, float3*& devOutputColor, float*& devOutputValidity, float*& devOutputDepth, float blendingExponent
);

#endif
