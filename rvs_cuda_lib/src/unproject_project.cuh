/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef UNPROJECT_PROJECT_CUH
#define UNPROJECT_PROJECT_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "types.cuh"


struct CamData
{
	float rotation[9];
	float translation[3];
	float focal[2];
	float principlePoint[2];
};

struct PrecomputedParams
{
	CamData camData;
	float devPhi0;
	float devTheta0;
	float dev_dphi_du;
	float dev_dtheta_dv;

	float devU0;
	float devV0;
	float dev_du_dphi;
	float dev_dv_dtheta;

	dim3 gridDim;
	dim3 blockDim;
};

template<typename channel_t>
using vec2_t = std::conditional_t<std::is_same<channel_t, float>::value, float2, double2>;

template<typename channel_t>
using vec3_t = std::conditional_t<std::is_same<channel_t, float>::value, float3, double3>;

template<typename position_t, typename channel_t>
void unprojectERP_projectERP(cv::Size size,
							 channel_t*& devDepth, position_t*& devTransformedPosition, channel_t*& devTransformedDepth,
							 const PrecomputedParams& params,
							 cudaStream_t& stream);

template<typename position_t, typename channel_t>
void unprojectERP_projectPerspective(cv::Size size,
									 channel_t*& devDepth, position_t*& devTransformedPosition, channel_t*& devTransformedDepth,
									 const PrecomputedParams& params,
									 cudaStream_t& stream);

#endif
