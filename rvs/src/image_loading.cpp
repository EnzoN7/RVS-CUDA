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

#include "image_loading.hpp"

namespace rvs
{
	namespace
	{
		using detail::ColorSpace;
		using detail::g_color_space;

		void read_raw(FILE* file, cv::Mat& image)
		{
			size_t read_count = fread(image.data, 1, image.total() * image.elemSize(), file);

			if (read_count != image.total() * image.elemSize())
			{
				std::cerr << "Failed to read the expected amount of data" << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}

		template<typename channel_t, typename color_t>
		void read_color_YUV(FILE*& inputColorFileYUV, int frame, color_t*& devNormalizedYUV,
							cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
							cudaStream_t& stream, cv::Mat& hostYUV,
							size_t yInputBytes, size_t uvInputBytes,
							void*& devReadYUV, cudaEvent_t& importColorFinished)
		{
			const long frameOffset = static_cast<long>((yInputBytes + 2 * uvInputBytes) * frame);

			fseek(inputColorFileYUV, frameOffset, SEEK_SET);
			read_raw(inputColorFileYUV, hostYUV);

			importColorsToGPU<channel_t, color_t>(hostYUV, devNormalizedYUV, initialY_size, realSize, initialColorsType, colorScale,
												  stream, yInputBytes, uvInputBytes,
												  devReadYUV, importColorFinished);
		}

		template<typename channel_t>
		void read_depth_YUV(FILE*& inputDepthFile, int frame, Parameters const& parameters, channel_t*& devNormalizedDepth,
							cv::Size initialDepth_size, cv::Size realSize, float depthScale, int initialDepthType,
							cudaStream_t& stream,
							cv::Mat& hostDepth,
							size_t inputDepthBytes,
							void*& devReadDepth,
							cudaEvent_t& importDepthFinished)
		{
			fseek(inputDepthFile, static_cast<long>((inputDepthBytes * 3 / 2) * frame), SEEK_SET);
			read_raw(inputDepthFile, hostDepth);

			float near = parameters.getDepthRange()[0];
			float far = parameters.getDepthRange()[1];
			bool hasInvalidDepth = parameters.hasInvalidDepth();

			importDepthToGPU<channel_t>(hostDepth, devNormalizedDepth, initialDepth_size, realSize, depthScale, near, far, hasInvalidDepth, initialDepthType, stream,
										inputDepthBytes,
										devReadDepth,
										importDepthFinished);
		}
	}

	template<typename channel_t, typename color_t>
	void read_color(FILE*& inputColorFileYUV, int frame, color_t*& devNormalizedYUV,
					cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
					cudaStream_t& stream,
					cv::Mat& hostYUV,
					size_t yInputBytes, size_t uvInputBytes,
					void*& devReadYUV, cudaEvent_t& importColorFinished)
	{
		read_color_YUV<channel_t, color_t>(inputColorFileYUV,
										   frame, devNormalizedYUV, initialY_size, realSize, colorScale, initialColorsType,
										   stream, hostYUV, yInputBytes, uvInputBytes,
										   devReadYUV, importColorFinished);
	}

	template<typename channel_t>
	void read_depth(FILE*& inputDepthFile, int frame, Parameters const& parameters, channel_t*& devNormalizedDepth,
					cv::Size initialDepth_size, cv::Size realSize, float depthScale, int initialDepthType,
					cudaStream_t& stream,
					cv::Mat& hostDepth,
					size_t inputDepthBytes,
					void*& devReadDepth,
					cudaEvent_t& importDepthFinished)
	{
		read_depth_YUV<channel_t>(inputDepthFile, frame, parameters, devNormalizedDepth, initialDepth_size, realSize, depthScale, initialDepthType, stream,
								  hostDepth, inputDepthBytes,
								  devReadDepth,
								  importDepthFinished);
	}

	template void read_color<float, float3>(
		FILE*& inputColorFileYUV, int frame, float3*& devNormalizedYUV,
		cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
		cudaStream_t& stream,
		cv::Mat& hostYUV,
		size_t yInputBytes, size_t uvInputBytes,
		void*& devReadYUV, cudaEvent_t& importColorFinished);

	template void read_color<double, double3>(
		FILE*& inputColorFileYUV, int frame, double3*& devNormalizedYUV,
		cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
		cudaStream_t& stream,
		cv::Mat& hostYUV,
		size_t yInputBytes, size_t uvInputBytes,
		void*& devReadYUV, cudaEvent_t& importColorFinished);

	template void read_color<half, half3>(
		FILE*& inputColorFileYUV, int frame, half3*& devNormalizedYUV,
		cv::Size initialY_size, cv::Size realSize, float colorScale, int initialColorsType,
		cudaStream_t& stream,
		cv::Mat& hostYUV,
		size_t yInputBytes, size_t uvInputBytes,
		void*& devReadYUV, cudaEvent_t& importColorFinished);

	template void read_depth<float>(
		FILE*&, int,
		Parameters const&,
		float*&,
		cv::Size, cv::Size,
		float, int,
		cudaStream_t&,
		cv::Mat&,
		size_t,
		void*&,
		cudaEvent_t&);

	template void read_depth<double>(
		FILE*&, int,
		Parameters const&,
		double*&,
		cv::Size, cv::Size,
		float, int,
		cudaStream_t&,
		cv::Mat&,
		size_t,
		void*&,
		cudaEvent_t&);

	template void read_depth<half>(
		FILE*&, int,
		Parameters const&,
		half*&,
		cv::Size, cv::Size,
		float, int,
		cudaStream_t&,
		cv::Mat&,
		size_t,
		void*&,
		cudaEvent_t&);
}
