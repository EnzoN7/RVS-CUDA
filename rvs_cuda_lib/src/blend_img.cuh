/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef BLEND_IMG_CUH
#define BLEND_IMG_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "types.cuh"

template<typename channel_t>
using vec3_t = std::conditional_t<std::is_same<channel_t, float>::value, float3, double3>;

template<typename channel_t>
struct Params
{
	float blendingExponent;
	int imgSize;
	int numImages;
	vec3_t<channel_t> emptyColor;
};

template<typename channel_t, typename color_t>
void blendImages(color_t** devColors, channel_t** devValidities, channel_t** devDepths, cv::Size outputSize,
				 color_t*& devBlendedColor, float blendingExponent, int numImages,
				 cudaStream_t& stream);

#endif
