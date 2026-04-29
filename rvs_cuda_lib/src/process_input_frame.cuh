#ifndef PROCESS_INPUT_FRAME_CUH
#define PROCESS_INPUT_FRAME_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void resizeImage(cv::Mat inputs[], float*& devUOutput, float*& devVOutput, cv::Size size, int type);

void mergeFrame(cv::Mat& yInput, float* devU, float* devV, float3*& devYUV, cv::Size outputSize, int yType, float colorScale);

void formatDepth(cv::Mat& src, float*& devDst, cv::Size size, float scale, float near, float far, bool hasInvalidDepth, int type);

void quantization(float3* devSrc, cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV, int rows, int cols, cv::Size outputSize, int cv_depth, unsigned max_val);

#endif
