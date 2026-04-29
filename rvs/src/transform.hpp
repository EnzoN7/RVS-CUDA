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
  Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be  Daniele Bonatto, Daniele.Bonatto@ulb.ac.be  Arnaud Schenkel, arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
  Bart Kroon, bart.kroon@philips.com
  Bart Sonneveldt, bart.sonneveldt@philips.com
*/

#ifndef _TRANSFORM_HPP_
#define _TRANSFORM_HPP_

#include <opencv2/core.hpp>

namespace rvs 
{
	namespace detail 
	{
		/**
		@file transform.hpp
		\brief The file containing the warping functions
		*/

		/**
		\brief Translate and rotate the camera in any 3D direction according to the input position map

		The image is divided into triangles which vertices are the centers of the pixels. Those triangle that are wrapped according to the input input position, and rasterized.

		The output is the new color image, the new corresponding depth map, and a quality map indicating the quality of each pixel.

		@param input_color Input color map
		@param input_depth Input depth map; valid values are > 0 (invalid may be NaN or <= 0)
		@param input_positions Warped coordinate map (result of unproject -> rotate/translate -> project)
		@param output_size Output size of the color image
		@param[out] depth Output depth map
		@param[out] quality Quality metric to drive blending; involves depth and shape of warped triangles (elongated and big = low quality)
		@param horizontalWrap
		@return Output color map
		*/
		cv::Mat3f transform_trianglesMethod(cv::Mat3f input_color, cv::Mat1f input_depth, cv::Mat2f input_positions, cv::Size output_size, cv::Mat1f& depth, cv::Mat1f& quality, bool horizontalWrap);
	}
}

#endif