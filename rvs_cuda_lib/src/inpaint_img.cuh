/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
    Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef INPAINT_IMG_CUH
#define INPAINT_IMG_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "types.cuh"

template<typename color_t>
void inpaintImg(color_t*& devColor, cv::Size dstSize, cudaStream_t& stream, ushort3*& devMap, bool*& devChange);

#endif