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

#include "image_writing.hpp"

namespace rvs
{
	namespace
	{
		using detail::ColorSpace;
		using detail::g_color_space;

		void write_raw(std::ofstream& outputFile, cv::Mat& component)
		{
			CV_Assert(outputFile.good() && !component.empty() && component.isContinuous());
			outputFile.write(reinterpret_cast<char const*>(component.data), component.size().area() * component.elemSize());
		}

		void write_color_YUV(std::ofstream& outputFile, cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV)
		{
			cudaEventSynchronize(writeY);
			write_raw(outputFile, dstY);

			cudaEventSynchronize(writeU);
			write_raw(outputFile, dstU);

			cudaEventSynchronize(writeV);
			write_raw(outputFile, dstV);
		}
	}

	template<typename channel_t, typename color_t>
	void write_color
	(
		std::ofstream& outputFile,
		color_t*& devColor, cv::Size virtualSize,
		cv::Size outputY_size, cv::Size outputUV_size, int outputType, unsigned outputMaxVal,
		cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
		cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
		cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV,
		size_t dstSizeY, size_t dstSizeUV,
		void*& devDstY, void*& devDstU, void*& devDstV
	)
	{
		exportColorsToCPU<channel_t, color_t>
		(
			devColor, dstY, dstU, dstV,
			virtualSize, outputY_size, outputUV_size, outputType, outputMaxVal,
			stream1, stream2, stream3,
			inpainted, exportedUV, writeY, writeU, writeV,
			dstSizeY, dstSizeUV,
			devDstY, devDstU, devDstV
		);

		write_color_YUV(outputFile, dstY, dstU, dstV, writeY, writeU, writeV);
	}

	template void write_color<float, float3>
	(
		std::ofstream& outputFile,
		float3*& devColor, cv::Size virtualSize,
		cv::Size outputY_size, cv::Size outputUV_size, int outputType, unsigned outputMaxVal,
		cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
		cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
		cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV,
		size_t dstSizeY, size_t dstSizeUV,
		void*& devDstY, void*& devDstU, void*& devDstV
	);

	template void write_color<double, double3>
		(
			std::ofstream& outputFile,
			double3*& devColor, cv::Size virtualSize,
			cv::Size outputY_size, cv::Size outputUV_size, int outputType, unsigned outputMaxVal,
			cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
			cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
			cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV,
			size_t dstSizeY, size_t dstSizeUV,
			void*& devDstY, void*& devDstU, void*& devDstV
		);

	template void write_color<half, half3>
	(
		std::ofstream& outputFile,
		half3*& devColor, cv::Size virtualSize,
		cv::Size outputY_size, cv::Size outputUV_size, int outputType, unsigned outputMaxVal,
		cudaStream_t& stream1, cudaStream_t& stream2, cudaStream_t& stream3,
		cudaEvent_t& inpainted, cudaEvent_t& exportedUV, cudaEvent_t& writeY, cudaEvent_t& writeU, cudaEvent_t& writeV,
		cv::Mat& dstY, cv::Mat& dstU, cv::Mat& dstV,
		size_t dstSizeY, size_t dstSizeUV,
		void*& devDstY, void*& devDstU, void*& devDstV
	);
}
