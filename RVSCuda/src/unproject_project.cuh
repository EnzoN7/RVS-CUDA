#ifndef UNPROJECT_PROJECT_CUH
#define UNPROJECT_PROJECT_CUH

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Auteur : Enzo DI MARIA
*/

void unprojectERP_projectERP
(
    cv::Size size, float* devDepth, float2*& devVirtualUV, float*& devVirtualDepth, cv::Vec2f hor_range, cv::Vec2f ver_range, cv::Matx33f R, cv::Vec3f t
);

void unprojectERP_projectPerspective
(
    cv::Size size, float* devDepth, float2*& devVirtualUV, float*& devVirtualDepth, cv::Vec2f hor_range, cv::Vec2f ver_range, cv::Matx33f R, cv::Vec3f t, cv::Vec2f f, cv::Vec2f p
);

#endif
