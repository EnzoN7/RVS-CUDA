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

#include "image_loading.hpp"
#include "Config.hpp"
#include "process_input_frame.cuh"

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>

namespace rvs
{
	namespace
	{
		using detail::ColorSpace;
		using detail::g_color_space;

		void read_raw(FILE* file, cv::Mat& image)
		{
			CV_Assert(file != nullptr && !image.empty() && image.isContinuous());

			size_t read_count = fread(image.data, 1, image.total() * image.elemSize(), file);

			if (read_count != image.total() * image.elemSize())
				throw std::runtime_error("Failed to read the expected amount of data");
		}

		float3* read_color_YUV(const char* filepath, int frame, Parameters const& parameters)
		{
			auto size = parameters.getSize();
			auto bit_depth = parameters.getColorBitDepth();
			auto type = CV_MAKETYPE(cvdepth_from_bit_depth(bit_depth), 1);
			cv::Mat y_channel(size, type);
			cv::Mat u_channel(size / 2, type);
			cv::Mat v_channel(size / 2, type);

			FILE* file;
			errno_t err = fopen_s(&file, filepath, "rb");
			if (err != 0)
			{
				std::ostringstream what;
				what << "Failed to read raw YUV color file \"" << filepath << "\"";
				throw std::runtime_error(what.str());
			}

			fseek(file, 0, SEEK_END);
			long fileSize = ftell(file);

			long offset = static_cast<long>(size.area() * y_channel.elemSize() * 3 / 2 * frame);

			if (offset >= fileSize)
				offset = static_cast<long>(0);

			fseek(file, offset, SEEK_SET);

			read_raw(file, y_channel);
			read_raw(file, u_channel);
			read_raw(file, v_channel);

			fclose(file);

			float* devU;
			float* devV;
			cv::Mat uv[] = { u_channel , v_channel };

			cv::Size newSize;
			if (g_with_cuda && detail::g_rescale != 1.0)
			{
				newSize = cv::Size(
					static_cast<int>(detail::g_rescale * size.width),
					static_cast<int>(detail::g_rescale * size.height));
			}
			else
			{
				newSize = size;
			}

			try
			{
				resizeImage(uv, devU, devV, newSize, type);
			}
			catch (const std::runtime_error& e)
			{
				std::cerr << "CUDA Error (resize) : " << e.what() << std::endl;
			}

			float3* yuv;
			static const float colorScale = 1.0f / max_level(parameters.getColorBitDepth());
			try
			{
				mergeFrame(y_channel, devU, devV, yuv, newSize, type, colorScale);
			}
			catch (const std::runtime_error& e)
			{
				std::cerr << "CUDA Error (merge) : " << e.what() << std::endl;
			}

			return yuv;
		}

		float* read_depth_YUV(const char* filepath, int frame, Parameters const& parameters)
		{
			auto size = parameters.getPaddedSize();
			auto bit_depth = parameters.getDepthBitDepth();
			auto type = CV_MAKETYPE(cvdepth_from_bit_depth(bit_depth), 1);
			cv::Mat image(size, type);

			FILE* file;
			errno_t err = fopen_s(&file, filepath, "rb");
			if (err != 0)
			{
				std::ostringstream what;
				what << "Failed to read raw YUV depth file \"" << filepath << "\"";
				throw std::runtime_error(what.str());
			}

			fseek(file, 0, SEEK_END);
			long fileSize = ftell(file);
			long offset;

			switch (parameters.getDepthColorFormat())
			{
				case ColorFormat::YUV420:
					offset = static_cast<long>(size.area() * image.elemSize() * 3 / 2 * frame);
					if (offset >= fileSize)
						offset = static_cast<long>(0);
					fseek(file, offset, SEEK_SET);
					break;

				case ColorFormat::YUV400:
					offset = static_cast<long>(size.area() * image.elemSize() * frame);
					if (offset >= fileSize)
						offset = static_cast<long>(0);
					fseek(file, offset, SEEK_SET);
					break;

				default:
					fclose(file);
					throw std::logic_error("Unknown depth map color format");
			}

			read_raw(file, image);

			fclose(file);

			float* depth;
			static const float depthScale = 1.0f / max_level(bit_depth);
			float near = parameters.getDepthRange()[0];
			float far = parameters.getDepthRange()[1];
			bool hasInvalidDepth = parameters.hasInvalidDepth();

			cv::Size newSize;
			if (g_with_cuda && detail::g_rescale != 1.0)
			{
				newSize = cv::Size(
					static_cast<int>(detail::g_rescale * size.width),
					static_cast<int>(detail::g_rescale * size.height));
			}
			else
			{
				newSize = size;
			}

			formatDepth(image, depth, newSize, depthScale, near, far, hasInvalidDepth, type);

			return depth;
		}
	}

	int cvdepth_from_bit_depth(int bit_depth)
	{
		if (bit_depth >= 1 && bit_depth <= 8)
			return CV_8U;
		else if (bit_depth >= 9 && bit_depth <= 16)
			return CV_16U;
		else if (bit_depth == 32)
			return CV_32F;
		else throw std::invalid_argument("invalid raw image bit depth");
	}

	unsigned max_level(int bit_depth)
	{
		assert(bit_depth > 0 && bit_depth <= 16);
		return (1u << bit_depth) - 1u;
	}

	float3* read_color(std::string filepath, int frame, Parameters const& parameters)
	{
		float3* yuv;
		if (filepath.substr(filepath.size() - 4, 4) == ".yuv")
		{
			yuv = read_color_YUV(filepath.c_str(), frame, parameters);
		}
		else
		{
			throw std::runtime_error("Readig multiple frames not (yet) supported for image files");
		}
		return yuv;
	}

	float* read_depth(std::string filepath, int frame, Parameters const& parameters)
	{
		float* image;
		if (filepath.substr(filepath.size() - 4, 4) == ".yuv")
		{
			image = read_depth_YUV(filepath.c_str(), frame, parameters);
		}
		else
		{
			throw std::runtime_error("Readig multiple frames not (yet) supported for image files");
		}

		return image;
	}

	PolynomialDepth read_polynomial_depth(std::string filepath, int frame, Parameters const& parameters)
	{
		// Load the image
		cv::Mat image;
		PolynomialDepth pd;
		std::string ext = filepath.substr(filepath.size() - 4, 4);
		if (frame == 0 && ext == ".exr") {
			std::array<cv::Mat1f,20> polynomial;
			for (int i = 0; i < 20; ++i) {
				std::stringstream ss;
				ss << i;
				size_t pos = filepath.find('*');
				std::string f = filepath.substr(0, pos) + ss.str() + filepath.substr(pos+1, filepath.size());
				cv::Mat1f p = cv::imread(f, cv::IMREAD_UNCHANGED);
			//	cv::GaussianBlur(p1,p1,cv::Size(11,11),3);
                polynomial[i] = p;
			}
			pd = PolynomialDepth(polynomial);
		}
		else if (ext == ".yuv") {
			std::array<cv::Mat1f, 20> polynomial;
			for (int i = 0; i < 20; ++i) {
				std::stringstream ss;
				ss << i;
				size_t pos = filepath.find('*');
				std::string f = filepath.substr(0, pos) + ss.str() + filepath.substr(pos + 1, filepath.size());
				//cv::Mat depth = read_depth_YUV(f.c_str(), frame, parameters);
				cv::Mat depth;
				cv::Mat1f p;
				depth.convertTo(p, CV_32F, 1. / max_level(parameters.getDepthBitDepth()));
				if (i != 9 && i != 19)
					p = (parameters.getMultiDepthRange()[1] - parameters.getMultiDepthRange()[0]) * p + parameters.getMultiDepthRange()[0];
				else if (i == 9)
					p = (parameters.getDepthRange()[0] + (parameters.getDepthRange()[1] - parameters.getDepthRange()[0]) * p) * parameters.getFocal()[0] / (parameters.getDepthRange()[1] * parameters.getDepthRange()[0]);
				polynomial[i] = p;
			}
			pd = PolynomialDepth(polynomial);
		}
		else {
			throw std::runtime_error("Readig multiple frames not (yet) supported for image files");
		}

		return pd;
	}
}
