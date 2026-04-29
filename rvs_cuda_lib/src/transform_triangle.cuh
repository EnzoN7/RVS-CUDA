#ifndef TRANSFORM_TRIANGLE_CUH
#define TRANSFORM_TRIANGLE_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Auteur : Enzo DI MARIA
*/

void transform_trianglesMethod
(
    float3* inputColor, cv::Size inputSize, float* devInputDepth, float2* devInputPositions, cv::Size outputSize, bool horizontalWrap,
    float3*& devOutputColor, float*& devOutputDepth, float*& devValidity
);

#endif
