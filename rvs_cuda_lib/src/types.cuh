#ifndef TYPES_CUH
#define TYPES_CUH

#include <iostream>
#include <cmath>
#include <cuda_fp16.h>


struct __align__(8) half3
{
	half2 Y;  // Luminance   Y + _padding
	half2 UV; // Chrominance U + V
};

#define hNAN __float2half(NAN) //

__device__ __forceinline__ float getY(const float3& pixel)
{
	return pixel.x;
}
__device__ __forceinline__ float getU(const float3& pixel)
{
	return pixel.y;
}
__device__ __forceinline__ float getV(const float3& pixel)
{
	return pixel.z;
}

__device__ __forceinline__ half2 getY(const half3& pixel)
{
	return pixel.Y;
}
__device__ __forceinline__ half2 getUV(const half3& pixel)
{
	return pixel.UV;
}

__device__ __forceinline__ bool isNaN(float x)
{
	return std::isnan(x);
}
__device__ __forceinline__ bool isNaN(double x)
{
	return std::isnan(x);
}
__device__ __forceinline__ bool isNaN(half x)
{
	return __hisnan(x);
}

__host__ __device__ __forceinline__ bool operator==(const float3& a, const float3& b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__host__ __device__ __forceinline__ bool operator==(const double3& a, const double3& b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__host__ __device__ __forceinline__ bool operator==(const half3& a, const half3& b)
{
	return (a.Y.x == b.Y.x) && (a.UV == b.UV);
}

__host__ __device__ __forceinline__ float3 operator*(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ double3 operator*(double a, double3 b)
{
	return make_double3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ __forceinline__ double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ double3 operator/(double3 a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}

#endif
