/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef SCALE_UV_CUH
#define SCALE_UV_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <iostream>

#include "types.cuh"

template<typename position_t>
void scaleUV(position_t*& devVirtualUV, float scale, int inputSize, cudaStream_t& stream);

#endif
