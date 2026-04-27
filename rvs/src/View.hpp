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

#ifndef _VIEW_HPP_
#define _VIEW_HPP_

#include "Parameters.hpp"
#include "image_loading.hpp"

#include <fstream>
#include <iostream>
#include <future>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace rvs
{
	class View
	{
	  public:
		virtual ~View() = default;

		virtual float get_max_depth() const
		{
			return 1.0;
		};

		virtual float get_min_depth() const
		{
			return 0.0;
		};

		virtual DisplacementMethod get_displacementMethod() const
		{
			return DisplacementMethod::depth;
		};

		double distance_from_origin = 1.0;

		std::string g_filename = "";
	};

	template<typename channel_t, typename color_t>
	class InputView : public View
	{
	  public:
		InputView(Config config, int view, int frame,
				  cv::Size initialSize, cv::Size realSize, cv::Size initialUV_size,
				  float colorScale, int initialColorsType, float depthScale, int initialDepthType,
				  size_t inputDepthBytes, size_t yInputBytes, size_t uvInputBytes, size_t outputBytes);

		~InputView();

		void load(cudaStream_t& streamCol, cudaStream_t& streamDep);

		float get_max_depth() const
		{
			return m_parameters.getDepthRange()[1];
		};

		float get_min_depth() const
		{
			return m_parameters.getDepthRange()[0];
		};

		// Sync GPU-side : enfile une dépendance sur le stream consommateur sans
		// bloquer le CPU (vs cudaEventSynchronize qui bloquait le thread appelant).
		color_t*& waitAndGetNormalizedColor(cudaStream_t& consumerStream)
		{
			cudaStreamWaitEvent(consumerStream, m_importColorFinished, 0);
			return m_devNormalizedYUV;
		}

		channel_t*& waitAndGetNormalizedDepth(cudaStream_t& consumerStream)
		{
			cudaStreamWaitEvent(consumerStream, m_importDepthFinished, 0);
			return m_devNormalizedDepth;
		}

		void setFrame(int frameToLoad)
		{
			m_frame = frameToLoad;
		}

		DisplacementMethod get_displacementMethod() const
		{
			return m_parameters.getDisplacementMethod();
		}

	  private:
		Config m_config;
		int m_view;

		const Parameters m_parameters;
		std::string m_filepath_color;
		std::string m_filepath_depth;

		FILE* m_inputColorFileYUV;
		FILE* m_inputDepthFile;

		int m_frame;
		float m_colorScale;
		float m_depthScale;
		int m_initialColorsType;
		int m_initialDepthType;

		cv::Size m_initialSize;
		cv::Size m_initialUV_size;
		cv::Size m_realSize;

		cv::Mat m_hostYUV;
		cv::Mat m_hostDepth;

		size_t m_inputDepthBytes;
		size_t m_yInputBytes;
		size_t m_uvInputBytes;

		void* m_devReadDepth;
		void* m_devReadYUV;

		color_t* m_devNormalizedYUV;
		channel_t* m_devNormalizedDepth;

		cudaEvent_t m_importColorFinished;
		cudaEvent_t m_importDepthFinished;

		void* m_pinned_host_yuv_ptr;
		void* m_pinned_host_depth_ptr;
	};
}

#endif
