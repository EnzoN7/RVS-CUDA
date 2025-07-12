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
  Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be
  Daniele Bonatto, Daniele.Bonatto@ulb.ac.be
  Arnaud Schenkel, arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
  Bart Kroon, bart.kroon@philips.com
  Bart Sonneveldt, bart.sonneveldt@philips.com
*/

/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#ifndef _BLENDED_VIEW_HPP_
#define _BLENDED_VIEW_HPP_

#include "SynthesizedView.hpp"
#include "View.hpp"
#include "blend_img.cuh"
#include "inpaint_img.cuh"
#include "image_writing.hpp"
#include "Config.hpp"

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <future>
#include <iostream>
#include <chrono>

namespace rvs
{
	template<typename channel_t, typename color_t>
	class BlendedView : public View
	{
	public:
		virtual ~BlendedView();
		virtual void blend(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3) = 0;
		virtual void setHostComponents(int viewIndex, color_t*& newColors, channel_t*& newValidities, channel_t*& newDepths) = 0;
		virtual void inpaint(cudaStream_t& stream) = 0;
		virtual void writeColor(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3) = 0;
	};

	template<typename channel_t, typename color_t>
	class BlendedViewSimple : public BlendedView<channel_t, color_t>
	{
	public:
		BlendedViewSimple
		(
			int numInputViews,
			cv::Size virtualSize, cv::Size outputY_size, cv::Size outputUV_size,
			int outputType, unsigned outputMaxVal, Config config
		);

		~BlendedViewSimple();

		void blend(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3);

		void inpaint(cudaStream_t& stream);

		void writeColor(cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3);

		void setHostComponents(int viewIndex, color_t*& newColors, channel_t*& newValidities, channel_t*& newDepths)
		{
			m_hostColorsAddr[viewIndex] = newColors;
			m_hostValiditiesAddr[viewIndex] = newValidities;
			m_hostDepthsAddr[viewIndex] = newDepths;
		}

	private:
		float m_blending_exp;
		int m_numInputViews;
		cv::Size m_virtualSize;

		color_t** m_hostColorsAddr;
		channel_t **m_hostValiditiesAddr;
		channel_t** m_hostDepthsAddr;

		color_t** m_devColorsAddr;
		channel_t** m_devValiditiesAddr;
		channel_t** m_devDepthsAddr;

		color_t* m_devBlendedColor;
		ushort3* m_devMap;
		ushort3* m_devMap_swap;
		int* m_devChange;

		cudaEvent_t m_importValiditiesAddr;
		cudaEvent_t m_importDepthsAddr;

		cudaEvent_t m_inpainted; 
		cudaEvent_t m_exportedUV;

		cudaEvent_t m_writeY;
		cudaEvent_t m_writeU;
		cudaEvent_t m_writeV;

		int m_virtualCam;

		cv::Size m_outputY_size;
		cv::Size m_outputUV_size;

		size_t m_dstSizeY;
		size_t m_dstSizeUV;

		cv::Mat m_dstY;
		cv::Mat m_dstU;
		cv::Mat m_dstV;

		int m_outputType;
		unsigned m_outputMaxVal;

		void* m_devDstY;
		void* m_devDstU;
		void* m_devDstV;

		void* m_pinned_host_Y_ptr;
		void* m_pinned_host_U_ptr;
		void* m_pinned_host_V_ptr;

		Config m_config;
		std::ofstream m_outputFile;
	};
}

#endif
