/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef IMPORT_EXPORT_IMAGES_CUH
#define IMPORT_EXPORT_IMAGES_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "types.cuh"

template<typename channel_t, typename color_t>
void importColorsToGPU(cv::Mat& hostYUV, color_t*& devNormalizedYUV,
					   cv::Size initialY_size, cv::Size realSize,
					   int type, float colorScale,
					   cudaStream_t& stream, size_t yuvInputBytes, size_t uvInputBytes,
					   void*& devReadYUV, cudaEvent_t& importColorFinished);

template<typename channel_t>
void importDepthToGPU(cv::Mat& hostDepth, channel_t*& devNormalizedDepth, cv::Size initialDepthSize, cv::Size realSize,
					  float scale, float near, float far, bool hasInvalidDepth, int type,
					  cudaStream_t& stream,
					  size_t inputDepthBytes,
					  void*& devReadDepth,
					  cudaEvent_t& importDepthFinished);

template<typename channel_t, typename color_t>
void exportColorsToCPU(color_t*& devYUV, cv::Mat& hostY, cv::Mat& hostU, cv::Mat& hostV,
					   cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
					   int cv_depth, unsigned max_val,
					   cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
					   cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
					   size_t dstSizeY, size_t dstSizeUV,
					   void*& devDstY, void*& devDstU, void*& devDstV);

#endif
