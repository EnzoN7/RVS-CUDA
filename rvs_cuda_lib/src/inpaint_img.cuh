#ifndef INPAINT_IMG_CUH
#define INPAINT_IMG_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Auteur : Enzo DI MARIA
*/

void inpaintImg(float3*& devColor, cv::Size dstSize);

#endif