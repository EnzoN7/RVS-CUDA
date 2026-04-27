/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "transform_triangle.cuh"


template<typename position_t>
__device__ __inline__ scalar_t<position_t> dot(position_t p1, position_t p2)
{
	scalar_t<position_t> dx = p1.x - p2.x;
	scalar_t<position_t> dy = p1.y - p2.y;
	return dx * dx + dy * dy;
}

template<typename position_t>
__device__ __inline__ scalar_t<position_t> isTriValid(position_t posA, position_t posB, position_t posC)
{
	scalar_t<position_t> ab = dot(posA, posB);
	scalar_t<position_t> ac = dot(posA, posC);
	scalar_t<position_t> bc = dot(posB, posC);

	scalar_t<position_t> mx = MAX(ab, MAX(ac, bc));

	scalar_t<position_t> qf;
	if constexpr (std::is_same<scalar_t<position_t>, float>::value)
	{
		qf = 10000.f - 1000.f * sqrtf(mx);
		qf = fminf(10000.f, fmaxf(1.f, qf));
	}
	else if constexpr (std::is_same<scalar_t<position_t>, double>::value)
	{
		qf = 10000.0 - 1000.0 * sqrt(mx);
		qf = fmin(10000.0, fmax(1.0, qf));
	}
	else if constexpr (std::is_same<scalar_t<position_t>, half>::value)
	{
		qf = __float2half(10000.f) - __float2half(1000.f) * hsqrt(mx);
		qf = MIN(__float2half(10000.f), MAX(__float2half(1.f), qf));
	}

	return qf;
}

template<typename channel_t>
struct DepthAndValidity
{
	channel_t depth;
	channel_t validity;
};

template<typename channel_t>
__device__ __inline__ encoded_t<channel_t> encodeDepthAndValidity(float depth, float validity)
{
	if constexpr (std::is_same<channel_t, float>::value)
	{
		union
		{
			DepthAndValidity<channel_t> dv;
			unsigned long long int e;
		} u;
		u.dv.depth = depth;
		u.dv.validity = validity;
		return u.e;
	}
	else if constexpr (std::is_same<channel_t, double>::value)
	{
		return {(double)depth, (double)validity};
	}
	else
	{
		return __floats2half2_rn(depth, validity);
	}
}

template<typename channel_t>
__device__ __inline__ DepthAndValidity<channel_t> decodeDepthAndValidity(encoded_t<channel_t> encoded)
{
	if constexpr (std::is_same<channel_t, float>::value)
	{
		union
		{
			DepthAndValidity<float> dv;
			unsigned long long int e;
		} u;
		u.e = encoded;
		return u.dv;
	}
	else if constexpr (std::is_same<channel_t, double>::value)
	{
		double2 h = encoded;
		DepthAndValidity<double> dv;
		dv.depth = h.x;
		dv.validity = h.y;
		return dv;
	}
	else
	{
		half2 h = encoded;
		DepthAndValidity<half> dv;
		dv.depth = h.x;
		dv.validity = h.y;
		return dv;
	}
}

template<typename channel_t>
__global__ void initializeArrayWithEncodedValues(encoded_t<channel_t>* array,
												 float depth,
												 float validity,
												 int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += stride)
		array[i] = encodeDepthAndValidity<channel_t>(depth, validity);
}

__global__ void initializeTriIdArray(int* array, int value, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += stride)
		array[i] = value;
}

template<typename channel_t>
__global__ void separateDepthAndValidityKernel(encoded_t<channel_t>* inputEnc,
											   channel_t* outDepth,
											   channel_t* outValid,
											   int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += stride)
	{
		DepthAndValidity<channel_t> dv = decodeDepthAndValidity<channel_t>(inputEnc[i]);
		outDepth[i] = dv.depth;
		outValid[i] = dv.validity;
	}
}

__device__ __inline__ void lockPixel(int* locks, int pixelId)
{
	while (atomicCAS(&locks[pixelId], 0, 1) != 0)
	{
	}
	__threadfence();
}

__device__ __inline__ void unlockPixel(int* locks, int pixelId)
{
	__threadfence();
	atomicExch(&locks[pixelId], 0);
}

template<
	typename position_t,
	typename color_t,
	typename channel_t>
__device__ void processPixel(int px, int py,
							 scalar_t<position_t> invArea,
							 color_t* outputColor,
							 encoded_t<channel_t>* outputDepthValidity,
							 int* outputColorLock,
							 int* outputTriId,
							 int triId,
							 scalar_t<position_t> triValid,
							 position_t posA, position_t posB, position_t posC,
							 color_t colA, color_t colB, color_t colC,
							 channel_t depA, channel_t depB, channel_t depC,
							 int outputWidth)
{
	channel_t fx = (channel_t)px + (channel_t)0.5 - posC.x;
	channel_t fy = (channel_t)py + (channel_t)0.5 - posC.y;

	channel_t l1 = invArea * ((posB.y - posC.y) * fx + (posC.x - posB.x) * fy);
	channel_t l2 = invArea * ((posC.y - posA.y) * fx + (posA.x - posC.x) * fy);
	channel_t l3 = (channel_t)1 - l1 - l2;

	if (l1 < (channel_t)1e-6 || l2 < (channel_t)1e-6 || l3 < (channel_t)1e-6)
		return;

	channel_t depth = depA * l1 + depB * l2 + depC * l3;

	vec3_t<channel_t> newCol =
		{
			colA.x * l1 + colB.x * l2 + colC.x * l3,
			colA.y * l1 + colB.y * l2 + colC.y * l3,
			colA.z * l1 + colB.z * l2 + colC.z * l3};

	int pid = py * outputWidth + px;

	encoded_t<channel_t> newEnc = encodeDepthAndValidity<channel_t>(depth, triValid);

	// Score scale-invariant : S = validity / depth^3 (triangle gagnant = max S).
	// Quantifié via __float_as_uint pour comparaison transitive (élimine la
	// non-associativité flottante). En cas d'égalité de qScore, le plus petit
	// triId gagne — rend le résultat indépendant de l'ordre du scheduler GPU.
	float depth_f = (float)depth;
	float scoreNew_f = (float)triValid / (depth_f * depth_f * depth_f);

	lockPixel(outputColorLock, pid);

	DepthAndValidity<channel_t> old = decodeDepthAndValidity<channel_t>(outputDepthValidity[pid]);
	int oldTriId = outputTriId[pid];

	float oldDepth_f = (float)old.depth;
	float scoreOld_f = (float)old.validity / (oldDepth_f * oldDepth_f * oldDepth_f);

	unsigned int qNew = __float_as_uint(scoreNew_f);
	unsigned int qOld = __float_as_uint(scoreOld_f);

	bool shouldUpdate = (qNew > qOld) || (qNew == qOld && triId < oldTriId);

	if (shouldUpdate)
	{
		outputColor[pid] = newCol;
		outputDepthValidity[pid] = newEnc;
		outputTriId[pid] = triId;
	}

	unlockPixel(outputColorLock, pid);
}

template<>
__device__ void processPixel<half2, half3, half>(int px, int py,
												 half invArea,
												 half3* outputColor,
												 half2* outputDepthValidity,
												 int* outputColorLock,
												 int* outputTriId,
												 int triId,
												 half triValid,
												 half2 posA, half2 posB, half2 posC,
												 half3 colA, half3 colB, half3 colC,
												 half depA, half depB, half depC,
												 int outputWidth)
{
	half fx = (half)px + __float2half(0.5f) - posC.x;
	half fy = (half)py + __float2half(0.5f) - posC.y;

	half l1 = invArea * ((posB.y - posC.y) * fx + (posC.x - posB.x) * fy);
	half l2 = invArea * ((posC.y - posA.y) * fx + (posA.x - posC.x) * fy);
	half l3 = (half)1 - l1 - l2;

	if (l1 < (half)6e-5 || l2 < (half)6e-5 || l3 < (half)6e-5)
		return;

	half depth = depA * l1 + depB * l2 + depC * l3;

	half2 lambda1_2 = make_half2(l1, l1);
	half2 lambda2_2 = make_half2(l2, l2);
	half2 lambda3_2 = make_half2(l3, l3);

	half3 newCol;
	newCol.Y = colA.Y * lambda1_2 + colB.Y * lambda2_2 + colC.Y * lambda3_2;
	newCol.UV = colA.UV * lambda1_2 + colB.UV * lambda2_2 + colC.UV * lambda3_2;

	int pid = py * outputWidth + px;

	half2 newEnc = encodeDepthAndValidity<half>(depth, triValid);

	float depth_f = __half2float(depth);
	float scoreNew_f = __half2float(triValid) / (depth_f * depth_f * depth_f);

	lockPixel(outputColorLock, pid);

	DepthAndValidity<half> old = decodeDepthAndValidity<half>(outputDepthValidity[pid]);
	int oldTriId = outputTriId[pid];

	float oldDepth_f = __half2float(old.depth);
	float scoreOld_f = __half2float(old.validity) / (oldDepth_f * oldDepth_f * oldDepth_f);

	unsigned int qNew = __float_as_uint(scoreNew_f);
	unsigned int qOld = __float_as_uint(scoreOld_f);

	bool shouldUpdate = (qNew > qOld) || (qNew == qOld && triId < oldTriId);

	if (shouldUpdate)
	{
		outputColor[pid] = newCol;
		outputDepthValidity[pid] = newEnc;
		outputTriId[pid] = triId;
	}

	unlockPixel(outputColorLock, pid);
}

template<
	typename position_t,
	typename color_t,
	typename channel_t>
__device__ void processTriangle(position_t posA, position_t posB, position_t posC,
								color_t colA, color_t colB, color_t colC,
								channel_t depA, channel_t depB, channel_t depC,
								color_t* outColor,
								encoded_t<channel_t>* outDepthValidity,
								int* locks,
								int* outputTriId,
								int triId,
								int outputWidth, int outputHeight)
{
	int xMin = MAX(0, (int)floorf(MIN(MIN(posA.x, posB.x), posC.x)));
	int yMin = MAX(0, (int)floorf(MIN(MIN(posA.y, posB.y), posC.y)));
	int xMax = MIN(outputWidth - 1, (int)ceilf(MAX(MAX(posA.x, posB.x), posC.x)));
	int yMax = MIN(outputHeight - 1, (int)ceilf(MAX(MAX(posA.y, posB.y), posC.y)));

	if (xMin >= xMax || yMin >= yMax)
		return;

	scalar_t<position_t> area, invArea, triV;
	if constexpr (std::is_same<scalar_t<position_t>, float>::value)
	{
		area = ((posB.y - posC.y) * (posA.x - posC.x) + (posC.x - posB.x) * (posA.y - posC.y));

		if (area <= 1e-6f)
			return;

		invArea = 1.0f / area;
		triV = isTriValid<float2>(posA, posB, posC);
	}
	else if constexpr (std::is_same<scalar_t<position_t>, double>::value)
	{
		area = ((posB.y - posC.y) * (posA.x - posC.x) + (posC.x - posB.x) * (posA.y - posC.y));

		if (area <= 1e-6)
			return;

		invArea = 1.0 / area;
		triV = isTriValid<double2>(posA, posB, posC);
	}
	else if constexpr (std::is_same<scalar_t<position_t>, half>::value)
	{
		area = ((posB.y - posC.y) * (posA.x - posC.x) + (posC.x - posB.x) * (posA.y - posC.y));

		if (__half2float(area) <= 6e-5f)
			return;

		invArea = __float2half(1.0f) / area;
		triV = isTriValid<half2>(posA, posB, posC);
	}

	for (int y = yMin; y <= yMax; ++y)
	{
		for (int x = xMin; x <= xMax; ++x)
		{
			processPixel<position_t, color_t, channel_t>(x, y,
														 invArea,
														 outColor,
														 outDepthValidity,
														 locks,
														 outputTriId,
														 triId,
														 triV,
														 posA, posB, posC,
														 colA, colB, colC,
														 depA, depB, depC,
														 outputWidth);
		}
	}
}

template<
	typename position_t,
	typename color_t,
	typename channel_t>
__global__ void colorizeTriangleUpLeftKernel(color_t* inputColor,
											 channel_t* inputDepth,
											 position_t* inputPos,
											 color_t* outputColor,
											 encoded_t<channel_t>* outputDepthValidity,
											 int* outputLock,
											 int* outputTriId,
											 int stride,
											 int inputWidth,
											 int inputHeight,
											 int outputWidth,
											 int outputHeight)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (; idx < inputWidth * inputHeight; idx += stride)
	{
		int tx = idx % inputWidth;
		int ty = idx / inputWidth;
		if (tx >= inputWidth - 1 || ty >= inputHeight - 1)
			continue;

		int idB = idx + 1;
		int idC = idx + inputWidth;

		channel_t dB = inputDepth[idB];
		channel_t dC = inputDepth[idC];
		if (isNaN(dB) || isNaN(dC))
			continue;

		color_t cB = inputColor[idB];
		color_t cC = inputColor[idC];
		position_t pB = inputPos[idB];
		position_t pC = inputPos[idC];

		channel_t dA = inputDepth[idx];
		if (isNaN(dA))
			continue;

		int triId = (idx << 2) | 0;
		processTriangle<position_t, color_t, channel_t>(inputPos[idx], pB, pC,
														inputColor[idx], cB, cC,
														dA, dB, dC,
														outputColor,
														outputDepthValidity,
														outputLock,
														outputTriId,
														triId,
														outputWidth,
														outputHeight);
	}
}

template<
	typename position_t,
	typename color_t,
	typename channel_t>
__global__ void colorizeTriangleDownRightKernel(color_t* inputColor,
												channel_t* inputDepth,
												position_t* inputPositions,
												color_t* outputColor,
												encoded_t<channel_t>* outputDepthValidity,
												int* outputColorLock,
												int* outputTriId,
												int stride,
												int inputWidth, int inputHeight,
												int outputWidth, int outputHeight)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (; idx < inputWidth * inputHeight; idx += stride)
	{
		int tx = idx % inputWidth;
		int ty = idx / inputWidth;
		if (tx <= 0 || ty <= 0)
			continue;

		int2 triPixelIds = make_int2(idx - inputWidth, idx - 1);
		channel_t dB = inputDepth[triPixelIds.x];
		channel_t dC = inputDepth[triPixelIds.y];

		if (isNaN(dB) || isNaN(dC))
			continue;

		color_t cB = inputColor[triPixelIds.x];
		color_t cC = inputColor[triPixelIds.y];
		position_t pB = inputPositions[triPixelIds.x];
		position_t pC = inputPositions[triPixelIds.y];

		channel_t dA = inputDepth[idx];
		if (isNaN(dA))
			continue;

		int triId = (idx << 2) | 1;
		processTriangle<position_t, color_t, channel_t>(inputPositions[idx], pC, pB,
														inputColor[idx], cC, cB,
														dA, dC, dB,
														outputColor, outputDepthValidity, outputColorLock,
														outputTriId, triId,
														outputWidth, outputHeight);
	}
}

template<
	typename position_t,
	typename color_t,
	typename channel_t>
__global__ void colorizeTriangleHorizontalWarpKernel(color_t* inputColor,
													 channel_t* inputDepth,
													 position_t* inputPositions,
													 color_t* outputColor,
													 encoded_t<channel_t>* outputDepthValidity,
													 int* outputColorLock,
													 int* outputTriId,
													 int stride,
													 int inputWidth, int inputHeight,
													 int outputWidth, int outputHeight)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (; idx < inputHeight; idx += stride)
	{
		int3 endTriPixelIds = make_int3(idx + inputWidth - 1,
										idx + 2 * inputWidth - 1,
										idx + inputWidth);
		channel_t dB = inputDepth[idx];
		channel_t dC = inputDepth[endTriPixelIds.y];

		if (isNaN(dB) || isNaN(dC))
			continue;

		color_t cB = inputColor[idx];
		color_t cC = inputColor[endTriPixelIds.y];
		position_t pB = inputPositions[idx];
		position_t pC = inputPositions[endTriPixelIds.y];
		channel_t dA = inputDepth[endTriPixelIds.x];

		if (!isNaN(dA))
		{
			int triId = (endTriPixelIds.x << 2) | 2;
			processTriangle<position_t, color_t, channel_t>(inputPositions[endTriPixelIds.x], pB, pC,
															inputColor[endTriPixelIds.x], cB, cC,
															dA, dB, dC,
															outputColor, outputDepthValidity, outputColorLock,
															outputTriId, triId,
															outputWidth, outputHeight);
		}

		dA = inputDepth[endTriPixelIds.z];

		if (!isNaN(dA))
		{
			int triId = (endTriPixelIds.z << 2) | 3;
			processTriangle<position_t, color_t, channel_t>(inputPositions[endTriPixelIds.z], pC, pB,
															inputColor[endTriPixelIds.z], cC, cB,
															dA, dC, dB,
															outputColor, outputDepthValidity, outputColorLock,
															outputTriId, triId,
															outputWidth, outputHeight);
		}
	}
}

template<typename channel_t>
void init(cv::Size virtualSize, int*& devColorLock, encoded_t<channel_t>*& devOutputDepthValidity,
		  int*& devOutputTriId,
		  cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& initDepthValidity)
{
	int outputImgSize = virtualSize.area();
	int blockSize = 128;
	int numBlocks = (outputImgSize - 1 + blockSize * 2) / (blockSize * 2);

	cudaMemsetAsync(devColorLock, 0, outputImgSize * sizeof(int), stream1);

	initializeArrayWithEncodedValues<channel_t><<<numBlocks, blockSize, 0, stream1>>>(devOutputDepthValidity, INFINITY, 0, outputImgSize);
	initializeTriIdArray<<<numBlocks, blockSize, 0, stream1>>>(devOutputTriId, 0x7FFFFFFF, outputImgSize);

	cudaEventRecord(initDepthValidity, stream1);
}

template<typename position_t,
		 typename color_t,
		 typename channel_t>
void synthesizeImageWithTrianglesMethod(
	color_t*& devInputColor, cv::Size inputSize, channel_t*& devInputDepth, position_t*& devInputPositions, cv::Size outputSize, bool horizontalWrap,
	color_t*& devOutputColor, channel_t*& devOutputDepth, channel_t*& devValidity,
	cudaStream_t& stream1, cudaStream_t& stream2,
	encoded_t<channel_t>*& devOutputDepthValidity, int*& devColorLock, int*& devOutputTriId,
	cudaEvent_t& synthesizeVirtualDepthValidity, cudaEvent_t& initDepthValidity, cudaEvent_t& colorizeTriangles, cudaEvent_t& projection,
	std::future<void>& futureProjection)
{
	int imgWidth = inputSize.width;
	int imgHeight = inputSize.height;

	int outputImgWidth = outputSize.width;
	int outputImgHeight = outputSize.height;
	int outputImgSize = outputSize.area();

	int blockSize = 256;
	int numBlocks = (imgHeight * imgWidth - 1 + blockSize * 2) / (blockSize * 2);
	int stride = blockSize * numBlocks;

	futureProjection.get();
	cudaStreamWaitEvent(stream1, projection, 0);

	(void)stream2;
	(void)initDepthValidity;
	(void)colorizeTriangles;

	colorizeTriangleUpLeftKernel<position_t, color_t, channel_t><<<numBlocks, blockSize, 0, stream1>>>(devInputColor,
																									   devInputDepth,
																									   devInputPositions,
																									   devOutputColor,
																									   devOutputDepthValidity,
																									   devColorLock,
																									   devOutputTriId,
																									   stride,
																									   imgWidth, imgHeight,
																									   outputImgWidth, outputImgHeight);

	colorizeTriangleDownRightKernel<position_t, color_t, channel_t><<<numBlocks, blockSize, 0, stream1>>>(devInputColor,
																										  devInputDepth,
																										  devInputPositions,
																										  devOutputColor,
																										  devOutputDepthValidity,
																										  devColorLock,
																										  devOutputTriId,
																										  stride,
																										  imgWidth, imgHeight,
																										  outputImgWidth, outputImgHeight);

	if (horizontalWrap)
	{
		numBlocks = (imgHeight - 1 + blockSize * 2) / (blockSize * 2);
		stride = blockSize * numBlocks;

		colorizeTriangleHorizontalWarpKernel<position_t, color_t, channel_t><<<numBlocks, blockSize, 0, stream1>>>(devInputColor,
																												   devInputDepth,
																												   devInputPositions,
																												   devOutputColor,
																												   devOutputDepthValidity,
																												   devColorLock,
																												   devOutputTriId,
																												   stride,
																												   imgWidth, imgHeight,
																												   outputImgWidth, outputImgHeight);
	}

	blockSize = 128;
	numBlocks = (outputImgSize - 1 + blockSize * 2) / (blockSize * 2);

	separateDepthAndValidityKernel<channel_t><<<numBlocks, blockSize, 0, stream1>>>(devOutputDepthValidity, devOutputDepth, devValidity, outputImgSize);
	cudaEventRecord(synthesizeVirtualDepthValidity, stream1);

#ifdef _DEBUG
	cudaError_t state = cudaGetLastError();
	if (state != cudaSuccess)
	{
		std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(state) << std::endl;
		std::exit(EXIT_FAILURE);
	}
#endif
}

template void init<float>(cv::Size virtualSize, int*& devColorLock, encoded_t<float>*& devOutputDepthValidity,
						  int*& devOutputTriId,
						  cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& initDepthValidity);

template void init<double>(cv::Size virtualSize, int*& devColorLock, encoded_t<double>*& devOutputDepthValidity,
						   int*& devOutputTriId,
						   cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& initDepthValidity);

template void init<half>(cv::Size virtualSize, int*& devColorLock, encoded_t<half>*& devOutputDepthValidity,
						 int*& devOutputTriId,
						 cudaStream_t& stream1, cudaStream_t& stream2, cudaEvent_t& initDepthValidity);

template void synthesizeImageWithTrianglesMethod<float2, float3, float>(
	float3*& devInputColor, cv::Size inputSize, float*& devInputDepth, float2*& devInputPositions, cv::Size outputSize, bool horizontalWrap,
	float3*& devOutputColor, float*& devOutputDepth, float*& devValidity,
	cudaStream_t& stream1, cudaStream_t& stream2,
	encoded_t<float>*& devOutputDepthValidity, int*& devColorLock, int*& devOutputTriId,
	cudaEvent_t& synthesizeVirtualDepthValidity, cudaEvent_t& initDepthValidity, cudaEvent_t& colorizeTriangles, cudaEvent_t& projection,
	std::future<void>& futureProjection);

template void synthesizeImageWithTrianglesMethod<double2, double3, double>(
	double3*& devInputColor, cv::Size inputSize, double*& devInputDepth, double2*& devInputPositions, cv::Size outputSize, bool horizontalWrap,
	double3*& devOutputColor, double*& devOutputDepth, double*& devValidity,
	cudaStream_t& stream1, cudaStream_t& stream2,
	encoded_t<double>*& devOutputDepthValidity, int*& devColorLock, int*& devOutputTriId,
	cudaEvent_t& synthesizeVirtualDepthValidity, cudaEvent_t& initDepthValidity, cudaEvent_t& colorizeTriangles, cudaEvent_t& projection,
	std::future<void>& futureProjection);

template void synthesizeImageWithTrianglesMethod<half2, half3, half>(
	half3*& devInputColor, cv::Size inputSize, half*& devInputDepth, half2*& devInputPositions, cv::Size outputSize, bool horizontalWrap,
	half3*& devOutputColor, half*& devOutputDepth, half*& devValidity,
	cudaStream_t& stream1, cudaStream_t& stream2,
	encoded_t<half>*& devOutputDepthValidity, int*& devColorLock, int*& devOutputTriId,
	cudaEvent_t& synthesizeVirtualDepthValidity, cudaEvent_t& initDepthValidity, cudaEvent_t& colorizeTriangles, cudaEvent_t& projection,
	std::future<void>& futureProjection);
