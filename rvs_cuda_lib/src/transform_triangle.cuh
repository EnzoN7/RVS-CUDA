/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef TRANSFORM_TRIANGLE_CUH
#define TRANSFORM_TRIANGLE_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <future>
#include <stdexcept>

#include "types.cuh"

template<typename VecT>
using scalar_t = decltype(VecT().x);

template<typename channel_t>
using encoded_t = std::conditional_t<std::is_same<channel_t, float>::value, unsigned long long int,
									 std::conditional_t<std::is_same<channel_t, double>::value, double2, half2>>;

template<typename channel_t>
using vec3_t = std::conditional_t<std::is_same<channel_t, float>::value, float3, double3>;


template<typename channel_t>
void init(cv::Size virtualSize, int*& devColorLock, encoded_t<channel_t>*& devOutputDepthValidity,
		  int*& devOutputTriId,
		  cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& initDepthValidity);

template<typename position_t,
		 typename color_t,
		 typename channel_t>
void synthesizeImageWithTrianglesMethod(
	color_t*& devInputColor, cv::Size inputSize, channel_t*& devInputDepth, position_t*& devInputPositions, cv::Size outputSize, bool horizontalWrap,
	color_t*& devOutputColor, channel_t*& devOutputDepth, channel_t*& devValidity,
	cudaStream_t& stream1, cudaStream_t& stream2,
	encoded_t<channel_t>*& devOutputDepthValidity, int*& devColorLock, int*& devOutputTriId,
	cudaEvent_t& synthesizeVirtualDepthValidity, cudaEvent_t& initDepthValidity, cudaEvent_t& colorizeTriangles, cudaEvent_t& projection,
	std::future<void>& futureProjection);

#endif
