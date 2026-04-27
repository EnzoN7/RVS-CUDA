/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "scale_uv.cuh"

template<typename position_t>
__global__ void scaleUVKernel(position_t* virtualUV, float scale, int inputSize)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < inputSize)
	{
		if constexpr (std::is_same<position_t, float2>::value)
		{
			virtualUV[threadId].x *= scale;
			virtualUV[threadId].y *= scale;
		}
		else if constexpr (std::is_same<position_t, double2>::value)
		{
			virtualUV[threadId].x *= (double)scale;
			virtualUV[threadId].y *= (double)scale;
		}
		else if constexpr (std::is_same<position_t, half2>::value)
		{
			virtualUV[threadId] *= __floats2half2_rn(scale, scale);
		}
	}
}

template<typename position_t>
void scaleUV(position_t*& devVirtualUV, float scale, int inputSize, cudaStream_t& stream)
{
	int threads_per_block = 128;

	int blocks_per_grid = (inputSize + threads_per_block - 1) / threads_per_block;

	scaleUVKernel<position_t><<<blocks_per_grid, threads_per_block, 0, stream>>>(devVirtualUV, scale, inputSize);

#ifdef _DEBUG
	cudaError_t state = cudaGetLastError();
	if (state != cudaSuccess)
	{
		std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
		std::exit(EXIT_FAILURE);
	}
#endif
}

template void scaleUV<float2>(float2*& devVirtualUV, float scale, int inputSize, cudaStream_t& stream);
template void scaleUV<double2>(double2*& devVirtualUV, float scale, int inputSize, cudaStream_t& stream);
template void scaleUV<half2>(half2*& devVirtualUV, float scale, int inputSize, cudaStream_t& stream);
