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

#ifndef _IMAGE_LOADING_HPP_
#define _IMAGE_LOADING_HPP_

#include "Parameters.hpp"
#include "Config.hpp"
#include "import_export_images.cuh"
#include "GpuDecoder.hpp"

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>

namespace rvs
{
	template<typename channel_t, typename color_t>
	void read_color(FILE*& inputColorFileYUV, int frame, color_t*& devNormalizedYUV,
		cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
		cudaStream_t& stream,
		cv::Mat& hostYUV,
		size_t yInputBytes, size_t uvInputBytes,
		void*& devReadYUV, cudaEvent_t& importColorFinished);

	template<typename channel_t, typename color_t>
	void read_color(GpuDecoder* decoder, int frame, color_t*& devNormalizedYUV,
		cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
		cudaStream_t& stream,
		cv::Mat& hostYUV,
		size_t yInputBytes, size_t uvInputBytes,
		void*& devReadYUV, cudaEvent_t& importColorFinished);

	template<typename channel_t>
	void read_depth(FILE*& inputDepthFile, int frame, Parameters const& parameters, channel_t*& devNormalizedDepth,
					cv::Size initialDepth_size, cv::Size realSize, float depthScale, int initialDepthType,
					cudaStream_t& stream,
					cv::Mat& hostDepth,
					size_t inputDepthBytes,
					void*& devReadDepth,
					cudaEvent_t& importDepthFinished);
}

#endif
