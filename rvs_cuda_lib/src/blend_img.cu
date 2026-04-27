#include "blend_img.cuh"

/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/


template<typename channel_t, typename color_t>
__global__ void blendImagesKernel(color_t** devColors, channel_t** devValidities, channel_t** devDepths,
								  color_t* outputColor, Params<channel_t> params)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t pxId = threadId; pxId < params.imgSize; pxId += stride)
	{
		channel_t sumWeights = (channel_t)0;
		channel_t inpaintedDepthSumWeights = (channel_t)0;

		vec3_t<channel_t> finalColor = {(channel_t)0, (channel_t)0, (channel_t)0};
		vec3_t<channel_t> inpaintedDepthColor = {(channel_t)0, (channel_t)0, (channel_t)0};

		for (int imId = 0; imId < params.numImages; imId++)
		{
			vec3_t<channel_t> color = devColors[imId][pxId];
			channel_t validity = devValidities[imId][pxId];
			channel_t depth = devDepths[imId][pxId];

			channel_t quality = validity / depth;

			if (quality > (channel_t)1e-6)
			{
				if (params.blendingExponent > (channel_t)1)
					quality = powf((float)quality, params.blendingExponent);

				if (quality > (channel_t)1e-6 && depth <= (channel_t)0)
				{
					sumWeights += quality;
					finalColor = finalColor + (quality * color);
				}
				else
				{
					inpaintedDepthSumWeights += quality;
					inpaintedDepthColor = inpaintedDepthColor + (quality * color);
				}
			}
		}

		if (sumWeights == (channel_t)0)
		{
			outputColor[pxId] = (inpaintedDepthSumWeights == (channel_t)0)
									? params.emptyColor
									: inpaintedDepthColor / inpaintedDepthSumWeights;
		}
		else
		{
			outputColor[pxId] = finalColor / sumWeights;
		}
	}
}

template<>
__global__ void blendImagesKernel<half, half3>(half3** devColors, half** devValidities, half** devDepths,
											   half3* outputColor, Params<half> params)
{
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t pxId = threadId; pxId < params.imgSize; pxId += stride)
	{
		half ZERO = __float2half(0.0f);
		half2 ZEROS = __floats2half2_rn(0.0f, 0.0f);

		half sumWeights = ZERO;
		half inpaintedDepthSumWeights = ZERO;

		half3 finalColor, inpaintedDepthColor;

		finalColor.Y = ZEROS;
		finalColor.UV = ZEROS;

		inpaintedDepthColor.Y = ZEROS;
		inpaintedDepthColor.UV = ZEROS;

		for (int imId = 0; imId < params.numImages; imId++)
		{
			half3 color = devColors[imId][pxId];
			half validity = devValidities[imId][pxId];
			half depth = devDepths[imId][pxId];

			half quality = validity / depth;

			if (quality > ZERO)
			{
				if (params.blendingExponent > 1.0f)
					quality = __float2half(powf(__half2float(quality), params.blendingExponent));

				half2 quality2 = __halves2half2(quality, quality);

				if (quality > ZERO && depth <= ZERO)
				{
					sumWeights = sumWeights + quality;
					finalColor.Y = finalColor.Y + quality2 * color.Y;
					finalColor.UV = finalColor.UV + quality2 * color.UV;
				}
				else
				{
					inpaintedDepthSumWeights = inpaintedDepthSumWeights + quality;
					inpaintedDepthColor.Y = inpaintedDepthColor.Y + quality2 * color.Y;
					inpaintedDepthColor.UV = inpaintedDepthColor.UV + quality2 * color.UV;
				}
			}
		}

		if (sumWeights == ZERO)
		{
			if (inpaintedDepthSumWeights == ZERO)
			{
				outputColor[pxId].Y = __floats2half2_rn((float)params.emptyColor.x, 0.0f);
				outputColor[pxId].UV = __floats2half2_rn((float)params.emptyColor.y, (float)params.emptyColor.z);
			}
			else
			{
				half2 weight2 = __halves2half2(inpaintedDepthSumWeights, inpaintedDepthSumWeights);

				outputColor[pxId].Y = inpaintedDepthColor.Y / weight2;
				outputColor[pxId].UV = inpaintedDepthColor.UV / weight2;
			}
		}
		else
		{
			half2 weight2 = __halves2half2(sumWeights, sumWeights);

			outputColor[pxId].Y = finalColor.Y / weight2;
			outputColor[pxId].UV = finalColor.UV / weight2;
		}
	}
}

template<typename channel_t, typename color_t>
void blendImages(color_t** devColors, channel_t** devValidities, channel_t** devDepths, cv::Size outputSize,
				 color_t*& devBlendedColor, float blendingExponent, int numImages,
				 cudaStream_t& stream)
{
	int imgHeight = outputSize.height;
	int imgWidth = outputSize.width;
	int imgSize = imgWidth * imgHeight;
	vec3_t<channel_t> emptyColor3 = {(channel_t)0, (channel_t)1, (channel_t)0};

	Params<channel_t> params = {blendingExponent, imgSize, numImages, emptyColor3};

	int blockSize = 512;
	int gridDim = (imgSize + blockSize - 1) / blockSize;

	blendImagesKernel<channel_t, color_t><<<gridDim, blockSize, 0, stream>>>(devColors, devValidities, devDepths,
																			 devBlendedColor, params);
}

template void blendImages<float, float3>(float3** devColors, float** devValidities, float** devDepths, cv::Size outputSize,
										 float3*& devBlendedColor, float blendingExponent, int numImages,
										 cudaStream_t& stream);

template void blendImages<double, double3>(double3** devColors, double** devValidities, double** devDepths, cv::Size outputSize,
										   double3*& devBlendedColor, float blendingExponent, int numImages,
										   cudaStream_t& stream);

template void blendImages<half, half3>(half3** devColors, half** devValidities, half** devDepths, cv::Size outputSize,
									   half3*& devBlendedColor, float blendingExponent, int numImages,
									   cudaStream_t& stream);
