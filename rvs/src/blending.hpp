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

#ifndef _BLENDING_HPP_
#define _BLENDING_HPP_

#include <opencv2/core.hpp>

/**
@file blending.hpp
\brief The file containing the image blending functions
*/

namespace rvs
{
	namespace detail
	{
		/**
		\brief Blend an array of color images chosing the best pixel.

		The best pixel is chosen among the input imgs: color(x,y)\f$=argmax_{c_i(x,y)}\left(w_i(x,y)\right)\f$, where \f$w_i\f$ is the quality of view i for the pixel (x,y).

		@param imgs Array of color images
		@param qualities Quality of the input image
		@param depth_prolongations Array of mask, indicating if the depth is valid or extrapolated
		@param empty_color Color for unknown pixels
		@param quality Output quality of the result
		@param depth_prolongation_mask Output mask indicating if the depth is valid or extrapolated
		@param inpaint_mask Output mask indicating where we have no value
		@return Color image containing the best pixels according to quality maps
		*/
		cv::Mat blend_img_by_max(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& qualities, const std::vector<cv::Mat>& depth_prolongations, cv::Vec3f empty_color, cv::Mat& quality, cv::Mat& depth_prolongation_mask, cv::Mat& inpaint_mask);

		/**
		 * \brief Split an image in low and high frequency.

		 * The low frequencies is obtained by applying a mean blur of size \f$\frac{1}{20}\f$ of the input image size. img=low_freq+high_freq
		 * @param img Image to process
		 * @param low_freq Output containing the low frequency part of the image
		 * @param high_freq Output containing the high frequency part of the image
		 * @param mask Mask for part of the image to ignore
		 * */
		void split_frequencies(const cv::Mat & img, cv::Mat& low_freq, cv::Mat& high_freq, const cv::Mat& mask);

		/**
		\brief Blend an array of color images by weigthed mean with quality of the pixels.

		Weighted mean: color(x,y)=\f$=\frac{\sum\limits_{i=0}^nw_i(x,y) c_i(x,y)}{\sum\limits_{i=0}^nw_i(x,y)}\f$ with \f$w_i(x,y)=\f$qualities[i](x,y)^blending_exp.

		If blending_exp < 0, calls blend_img_by_max().
		@param imgs Array of color images to blend
		@param qualities Array quality maps
		@param depth_prolongations Array of masks, indicating if the depth is valid or extrapolated
		@param empty_color Color for unknown pixels
		@param quality Output quality of the result
		@param depth_prolongation_mask Output mask indicating if the depth is valid or extrapolated
		@param inpaint_mask Output mask indicating where we have no value
		@param blending_exp
		<0 : Calls blend_img_by_max().
		0 : Mean of the available RGB images.
		>0 : Mean weighted by quality^blending_exp.
		@return Color image containing the the weighted mean of the input images.
		*/
		cv::Mat blend_img(const std::vector<cv::Mat> colors, const std::vector<cv::Mat> qualities, const std::vector<cv::Mat> depthProlongations, cv::Vec3f emptyColor, cv::Mat& outputQuality, cv::Mat& outputDepthProlongationMask, cv::Mat& outputInpaintMask, float blendingExponent);
	}
}

#endif