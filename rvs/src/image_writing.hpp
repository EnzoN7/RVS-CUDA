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

#ifndef _IMAGE_WRITING_HPP_
#define _IMAGE_WRITING_HPP_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Parameters.hpp"

namespace rvs
{
	/**
	@file image_writing.hpp
	\brief The file containing the image writing functions
	*/

	/**
	\brief Write a color image in RGB or YUV 4:2:0 fileformat.

	@param filepath Name of the image file to write
	@param color Image to write
	@param frame Frame number (for YUV)
	@param parameters Camera and video parameters
	*/
	void write_color(std::string filepath, cv::Mat3f color, int frame, Parameters const& parameters);
	void write_color(std::string filepath, float3* devColor, int frame, Parameters const& parameters, cv::Size);

	/**
	\brief Write a depth map in RGB or YUV 4:2:0 fileformat.

	@param filepath Name of the image file to write
	@param depth Image to write
	@param frame Frame number (for YUV)
	@param parameters Camera and video parameters
	*/
	void write_depth(std::string filepath, cv::Mat1f depth, int frame, Parameters const& parameters);

	/**
	\brief Write a masked depth map in RGB or YUV 4:2:0 fileformat.

	@param filepath Name of the image file to write
	@param depth Image to write
	@param mask Binary mask to apply before writing
	@param frame Frame number (for YUV)
	@param parameters Camera and video parameters
	*/
	void write_maskedDepth(std::string filepath, cv::Mat1f depth, cv::Mat1b mask, int frame, Parameters const& parameters);

	/**
	\brief Write a mask in RGB or YUV 4:2:0 fileformat.

	@param filepath Name of the image file to write
	@param depth Image to write
	@param frame Frame number (for YUV)
	@param parameters Camera and video parameters
	*/
	void write_mask(std::string filepath, cv::Mat1b mask, int frame, Parameters const& parameters);
}

#endif