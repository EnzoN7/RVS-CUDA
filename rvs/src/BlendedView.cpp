/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2018, ITU/ISO/IEC
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
Original authors:

Universite Libre de Bruxelles, Brussels, Belgium:
	Sarah Fachada Sarah.Fernandes.Pinto.Fachada@ulb.ac.be
	Daniele Bonatto Daniele.Bonatto@ulb.ac.be
	Arnaud Schenkel arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
	Bart Kroon, bart.kroon@philips.com
	Bart Sonneveldt, bart.sonneveldt@philips.com
*/

/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#include "BlendedView.hpp"

namespace rvs
{
	template<typename channel_t, typename color_t>
	BlendedView<channel_t, color_t>::~BlendedView() {}

	template<typename channel_t, typename color_t>
	BlendedViewSimple<channel_t, color_t>::BlendedViewSimple
	(
		int numInputViews,
		cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
		int outputType, unsigned outputMaxVal, Config config
	)
		: m_blending_exp(config.blending_factor)
		, m_numInputViews(numInputViews)
		, m_virtualSize(virtualSize)
		, m_virtualCam(0)
		, m_outputY_size(outputY_size)
		, m_outputUV_size(outputUV_size)
		, m_outputType(outputType)
		, m_outputMaxVal(outputMaxVal)
		, m_config(config)
		, m_outputFile(config.outfilenames[0], std::ios::binary)
	{
		m_dstSizeY = m_outputY_size.area() * CV_ELEM_SIZE(m_outputType);
		m_dstSizeUV = m_outputUV_size.area() * CV_ELEM_SIZE(m_outputType);

		m_hostColorsAddr = new color_t*[m_numInputViews];
		m_hostValiditiesAddr = new channel_t*[m_numInputViews];
		m_hostDepthsAddr = new channel_t*[m_numInputViews];

		cudaMallocHost(&m_pinned_host_Y_ptr, m_dstSizeY);
		cudaMallocHost(&m_pinned_host_U_ptr, m_dstSizeUV);
		cudaMallocHost(&m_pinned_host_V_ptr, m_dstSizeUV);

		m_dstY = cv::Mat(outputY_size, outputType, m_pinned_host_Y_ptr);
		m_dstU = cv::Mat(outputUV_size, outputType, m_pinned_host_U_ptr);
		m_dstV = cv::Mat(outputUV_size, outputType, m_pinned_host_V_ptr);

		cudaMalloc(&m_devColorsAddr, m_numInputViews * sizeof(color_t*));
		cudaMalloc(&m_devValiditiesAddr, m_numInputViews * sizeof(channel_t*));
		cudaMalloc(&m_devDepthsAddr, m_numInputViews * sizeof(channel_t*));

		cudaMalloc(&m_devBlendedColor, m_virtualSize.area() * sizeof(color_t));

		cudaMalloc(&m_devMap, m_virtualSize.area() * sizeof(ushort3));
		cudaMalloc(&m_devMap_swap, m_virtualSize.area() * sizeof(ushort3));
		cudaMalloc(&m_devChange, sizeof(int));

		cudaMalloc(&m_devDstY, m_dstSizeY);
		cudaMalloc(&m_devDstU, m_dstSizeUV);
		cudaMalloc(&m_devDstV, m_dstSizeUV);

		cudaEventCreateWithFlags(&m_importValiditiesAddr, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_importDepthsAddr, cudaEventDisableTiming);

		cudaEventCreateWithFlags(&m_inpainted, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_exportedUV, cudaEventDisableTiming);

		cudaEventCreateWithFlags(&m_writeY, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_writeU, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_writeV, cudaEventDisableTiming);

#ifdef _DEBUG
		cudaError_t state = cudaGetLastError();
		if (state != cudaSuccess)
		{
			std::cerr << "[cudaMalloc BlendedViewSimple ERROR]: " << cudaGetErrorString(state) << std::endl;
			std::exit(EXIT_FAILURE);
		}
#endif
	}

	template<typename channel_t, typename color_t>
	BlendedViewSimple<channel_t, color_t>::~BlendedViewSimple()
	{
		cudaFree(m_devBlendedColor);

		cudaFreeHost(m_pinned_host_Y_ptr);
		cudaFreeHost(m_pinned_host_U_ptr);
		cudaFreeHost(m_pinned_host_V_ptr);

		cudaFree(m_devColorsAddr);
		cudaFree(m_devValiditiesAddr);
		cudaFree(m_devDepthsAddr);

		cudaFree(m_devMap);
		cudaFree(m_devMap_swap);
		cudaFree(m_devChange);

		delete[] m_hostColorsAddr;
		delete[] m_hostValiditiesAddr;
		delete[] m_hostDepthsAddr;

		cudaEventDestroy(m_importValiditiesAddr);
		cudaEventDestroy(m_importDepthsAddr);

		cudaEventDestroy(m_inpainted);
		cudaEventDestroy(m_exportedUV);

		cudaEventDestroy(m_writeY);
		cudaEventDestroy(m_writeU);
		cudaEventDestroy(m_writeV);

		cudaFree(m_devDstU);
		cudaFree(m_devDstV);
		cudaFree(m_devDstY);

		m_dstY.release();
		m_dstU.release();
		m_dstV.release();

		m_outputFile.close();

#ifdef _DEBUG
		cudaError_t state = cudaGetLastError();
		if (state != cudaSuccess)
		{
			std::cerr << "[cudaFree BlendedViewSimple ERROR]: " << cudaGetErrorString(state) << std::endl;
			std::exit(EXIT_FAILURE);
		}
#endif
	}

	template<typename channel_t, typename color_t>
	void BlendedViewSimple<channel_t, color_t>::blend(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3)
	{
		cudaMemcpyAsync(m_devColorsAddr, m_hostColorsAddr, m_numInputViews * sizeof(color_t*), cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(m_devValiditiesAddr, m_hostValiditiesAddr, m_numInputViews * sizeof(channel_t*), cudaMemcpyHostToDevice, stream2);
		cudaEventRecord(m_importValiditiesAddr, stream2);

		cudaMemcpyAsync(m_devDepthsAddr, m_hostDepthsAddr, m_numInputViews * sizeof(channel_t*), cudaMemcpyHostToDevice, stream3);
		cudaEventRecord(m_importDepthsAddr, stream3);

		cudaStreamWaitEvent(stream1, m_importValiditiesAddr, 0);
		cudaStreamWaitEvent(stream1, m_importDepthsAddr, 0);

		blendImages<channel_t, color_t>(m_devColorsAddr, m_devValiditiesAddr, m_devDepthsAddr,
										m_virtualSize, m_devBlendedColor, m_blending_exp, m_numInputViews,
										stream1);

#ifdef _DEBUG
		cudaError_t state = cudaGetLastError();
		if (state != cudaSuccess)
		{
			std::cerr << "[BlendedViewSimple::blend ERROR]: " << cudaGetErrorString(state) << std::endl;
			std::exit(EXIT_FAILURE);
		}
#endif
	}

	template<typename channel_t, typename color_t>
	void BlendedViewSimple<channel_t, color_t>::inpaint(cudaStream_t& stream)
	{
		inpaintImg<color_t>(m_devBlendedColor, m_virtualSize, stream, m_devMap, m_devMap_swap, m_devChange);
		cudaEventRecord(m_inpainted, stream);
	}

	template<typename channel_t, typename color_t>
	void BlendedViewSimple<channel_t, color_t>::writeColor(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3)
	{
		write_color<channel_t, color_t>(m_outputFile,
										m_devBlendedColor,
										m_virtualSize, m_outputY_size, m_outputUV_size,
										m_outputType, m_outputMaxVal,
										stream1, stream2, stream3,
										m_inpainted, m_exportedUV, m_writeY, m_writeU, m_writeV,
										m_dstY, m_dstU, m_dstV,
										m_dstSizeY, m_dstSizeUV,
										m_devDstY, m_devDstU, m_devDstV);
	}

	template class BlendedView<float, float3>;
	template class BlendedView<double, double3>;
	template class BlendedView<half, half3>;

	template class BlendedViewSimple<float, float3>;
	template class BlendedViewSimple<double, double3>;
	template class BlendedViewSimple<half, half3>;
}
