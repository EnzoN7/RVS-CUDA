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

#ifndef _IMAGE_LOADING_HPP_
#define _IMAGE_LOADING_HPP_

#include "Parameters.hpp"
#include "PolynomialDepth.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace rvs
{
	/**
	@file image_loading.hpp
	\brief The file containing the image loading functions
	*/

	/**
	\brief Return the depth needed to encode an image according to its bit depth.

	\exception std::invalid_argument("invalid raw image bit depth")
	@param bit_depth Bit depth
	@return CV_8U for a bit depth between 1 and 8, CV_16U for a bit depth between 9 and 16: CV_16U
	*/
	int cvdepth_from_bit_depth(int bit_depth);

	/**
	\brief Return the maximum level corresponding to this bit depth.

	@param bit_depth Bit depth
	@return (1u << bit_depth) - 1u
	*/
	unsigned max_level(int bit_depth);

	/**
	\brief Read a color image (RGB or YUV).

	Use openCV cv::imread() function to read non .YUV images.
	@param filepath Name of the image file (YUV, PNG, etc.)
	@param frame Number of the frame to read
	@param parameters Camera and video parameters
	@return CV_32FC3 image
	*/
	float3* read_color(std::string filepath, int frame, Parameters const& parameters);

	/**
	\brief Read a depth image: a exr depth file or a YUV disparity file.

	Use openCV cv::imread() function to read non .YUV images.
	@param filepath Name of the image file (YUV, PNG, etc.)
	@param frame Number of the frame to read
	@param parameters Camera and video parameters
	@return CV_32F image

	Result may have NaN values to indicate missing depth values
	*/
	float* read_depth(std::string filepath, int frame, Parameters const& parameters);
	rvs::PolynomialDepth read_polynomial_depth(std::string filepath, int frame, Parameters const& parameters);
}

#endif
