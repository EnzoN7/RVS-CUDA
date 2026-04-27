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

#include "View.hpp"

#include "image_writing.hpp"

namespace rvs
{
	template<typename channel_t, typename color_t>
	InputView<channel_t,color_t>::InputView(Config config, int view, int frame,
		cv::Size initialSize, cv::Size realSize, cv::Size initialUV_size,
		float colorScale, int initialColorsType, float depthScale, int initialDepthType,
		size_t inputDepthBytes, size_t yInputBytes, size_t uvInputBytes, size_t outputBytes)
		: m_config(config)
		, m_view(view)
		, m_parameters(config.params_real[view])
		, m_filepath_color(config.texture_names[view])
		, m_filepath_depth(config.depth_names[view])
		, m_frame(frame)
		, m_colorScale(colorScale)
		, m_depthScale(depthScale)
		, m_initialColorsType(initialColorsType)
		, m_initialDepthType(initialDepthType)
		, m_initialSize(initialSize)
		, m_realSize(realSize)
		, m_initialUV_size(initialUV_size)
		, m_inputDepthBytes(inputDepthBytes)
		, m_yInputBytes(yInputBytes)
		, m_uvInputBytes(uvInputBytes)
	{
		const int totalPx = initialSize.area() + 2 * initialUV_size.area();
		const size_t yuv_buffer_size = totalPx * CV_ELEM_SIZE(initialColorsType);
		const size_t depth_buffer_size = initialSize.area() * CV_ELEM_SIZE(initialDepthType);

		cudaMallocHost(&m_pinned_host_yuv_ptr, yuv_buffer_size);
		cudaMallocHost(&m_pinned_host_depth_ptr, depth_buffer_size);

		m_hostYUV = cv::Mat(1, totalPx, initialColorsType, m_pinned_host_yuv_ptr);
		m_hostDepth = cv::Mat(initialSize, initialDepthType, m_pinned_host_depth_ptr);

		cudaMalloc(&m_devReadDepth, inputDepthBytes);
		cudaMalloc(&m_devReadYUV, yInputBytes + 2 * uvInputBytes);

		cudaMalloc(&m_devNormalizedYUV, m_realSize.area() * sizeof(color_t));
		cudaMalloc(&m_devNormalizedDepth, outputBytes);

		cudaEventCreateWithFlags(&m_importColorFinished, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_importDepthFinished, cudaEventDisableTiming);

		errno_t err = fopen_s(&m_inputColorFileYUV, m_filepath_color.c_str(), "rb");
		if (err != 0)
		{
			std::cerr << "[FILE ERROR]: " << "Failed to read raw YUV depth file \"" << m_filepath_color.c_str() << "\"" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		err = fopen_s(&m_inputDepthFile, m_filepath_depth.c_str(), "rb");
		if (err != 0)
		{
			std::cerr << "[FILE ERROR]: " << "Failed to read raw YUV depth file \"" << m_filepath_depth.c_str() << "\"" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	template<typename channel_t, typename color_t>
	InputView<channel_t, color_t>::~InputView()
	{
		cudaFreeHost(m_pinned_host_yuv_ptr);
		cudaFreeHost(m_pinned_host_depth_ptr);

		cudaFree(m_devReadDepth);
		cudaFree(m_devReadYUV);
		cudaFree(m_devNormalizedYUV);
		cudaFree(m_devNormalizedDepth);

		cudaEventDestroy(m_importColorFinished);
		cudaEventDestroy(m_importDepthFinished);

		m_hostYUV.release();
		m_hostDepth.release();

		fclose(m_inputColorFileYUV);
		fclose(m_inputDepthFile);
	}

	template<typename channel_t, typename color_t>
	void InputView<channel_t, color_t>::load(cudaStream_t& streamCol, cudaStream_t& streamDep)
	{
		if (m_parameters.getDisplacementMethod() == DisplacementMethod::depth && m_parameters.getDepthColorFormat() == ColorFormat::YUV420)
		{
			std::future<void> future_color = std::async(std::launch::async,
				[this, &streamCol]()
				{
					read_color<channel_t, color_t>(m_inputColorFileYUV,
												   m_frame, m_devNormalizedYUV, m_initialSize, m_realSize,
												   m_colorScale, m_initialColorsType,
												   streamCol,
												   m_hostYUV,
												   m_yInputBytes, m_uvInputBytes,
												   m_devReadYUV,
												   m_importColorFinished);
				}
			);

			std::future<void> future_depth = std::async(std::launch::async,
				[this, &streamDep]()
				{
					read_depth<channel_t>(m_inputDepthFile, m_frame, m_parameters, m_devNormalizedDepth, m_initialSize, m_realSize,
										  m_depthScale, m_initialDepthType, streamDep,
										  m_hostDepth,
										  m_inputDepthBytes,
										  m_devReadDepth,
										  m_importDepthFinished);
				}
			);

			future_color.get();
			future_depth.get();
		}
		else
		{
			std::cerr << "[ERROR] Unknown color format." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	template class InputView<float, float3>;
	template class InputView<double, double3>;
	template class InputView<half, half3>;
}
