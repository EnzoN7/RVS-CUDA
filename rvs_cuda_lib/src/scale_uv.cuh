#ifndef SCALE_UV_CUH
#define SCALE_UV_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>

/*
* Auteur : Enzo DI MARIA
*/

void scaleUV(float2*& devVirtualUV, float scale, int inputSize);

#endif
