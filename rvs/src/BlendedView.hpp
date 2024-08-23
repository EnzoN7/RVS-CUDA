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

#ifndef _BLENDED_VIEW_HPP_
#define _BLENDED_VIEW_HPP_

#include "SynthesizedView.hpp"
#include "View.hpp"

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
@file BlendedView.hpp
*/

namespace rvs
{
	class View;
	/**
	\brief Blending class: blends synthetised views one by one as they are generated.

	The blending is done by comparing a per-pixel quality, determined by synthesized depth
	maps and the shape of the triangle a pixel lies in: a pixel of good quality has a low
	depth (e.g. in the foreground) and belongs to a triangle with a regular shape (e.g.
	does not lie in a disocclusion).

	Taking the pixel with the maximal quality would give a sharper result, while taking
	the weighted mean is more resistant to errors in the depth maps, color differences
	between the input views and, in the case of navigation leads to smoother navigation,
	with less flickering (BlendedViewSimple). High and low frequencies can be separated with
	a mean blur with a kernel of size of 10% of the image size, and the most adapted
	blending can be applied to each frequency (BlendedViewMultiSpec).
	*/
	class BlendedView : public View
	{
	public:
		virtual ~BlendedView();

		/**
		\brief Blends this view with a new view.
		*/
		virtual void blend(View const& view) = 0;

		virtual void freeMemory(cv::Size imgSize) = 0;
#if WITH_OPENGL
		/**
		\brief Transfert textures from OpenGL to OpenCV Matrices (color, validity)
		*/
		void assignFromGL2CV(cv::Size size);
#endif
	};

	/**
	\brief Simple blending

	The final color \f$c\f$
	of a pixel is:

	\f$c=\frac{\sum\limits_{i=0}^nw_i c_i}{\sum\limits_{i=0}^nw_i}\f$ with \f$w_i=\left(\frac{q_i}{d_i}\right)^\alpha\f$ if \f$\alpha\geq 0\f$

	or \f$c=argmax_{c_i}\left(w_i\right)\f$ if \f$\alpha<0\f$

	where \f$n\f$ is the number of input views, \f$c_i\f$ the color of the pixel in view \f$i\f$, \f$q_i\f$ the quality of the triangle
	the pixel lies in (see transform_trianglesMethod()),
	\f$\alpha\f$ the blending_exp parameter and \f$d_i\f$ the depth at this pixel for the synthesized view \f$i\f$.

	We have
	\f$
	q_i=2\mathcal{A}/b^2\f$ if the triangle keeps the same orientation during warping, 0 otherwise. where \f$\mathcal{A}\f$ is the area
	of the triangle, \f$b\f$ is the length of the second longest side of the triangle.
	*/
	class BlendedViewSimple : public BlendedView {
	public:
		/**
		\brief Initialise an empty blending

		\param blending_exp \f$a\f$ in the formula \f$color=(\sum_i quality_i^a*color_i)/(\sum_i quality_i)\f$ or \f$color=color_{argmax(quality_i)}\f$ if \f$a<0\f$.
		*/
		BlendedViewSimple(float blending_exp);

		/**
		\brief Destructor
		*/
		~BlendedViewSimple();

		/**
		\brief Adds a new view to the blended image.

		The resulting image has the per-pixel value: \f$color=(\sum_i quality_i^a*color_i)/(\sum_i quality_i)\f$ or \f$color=color_{argmax(quality_i)}\f$ if \f$a<0\f$, where \f$a\f$ is the blending exponent.
		@param view View to add to the blended image.
		*/
		void blend(View const& view);
		void blendCudaMode(View const& view);
		void blendBasicMode(View const& view);

		void freeMemory(cv::Size imgSize);

	private:
		bool m_is_empty;

		/** The value of a in the formula \f$color=(\sum_i quality_i^a*color_i)/(\sum_i quality_i)\f$ or \f$color=color_{argmax(quality_i)}\f$ if \f$a<0\f$ */
		float m_blending_exp;

		// Depth mask
		cv::Mat1b m_depth_mask;
	};

	/**
	\brief Multispectral blending: this algorithm divides the input images in a low frequency image and a high frequency image, to blend them with different coefficients (exp_high_freq and exp_low_freq) with BlendedViewSimple algorithm.

	The final color \f$c\f$
	of a pixel is hence:
	\f$c=c_{high}+c_{low}\f$

	where \f$c_{high}\f$, \f$c_{low}\f$ are the resulting colors of high and low frequencies simple blending.
	*/
	class BlendedViewMultiSpec : public BlendedView {
	public:
		/**
		\brief Initialise an empty blending
		@param exp_low_freq Blending factor for the low frequencies
		@param exp_high_freq Blending factor for the high frequencies
		*/
		BlendedViewMultiSpec(float exp_low_freq, float exp_high_freq);
		/**
		\brief Destructor
		*/
		~BlendedViewMultiSpec();

		/**
		\brief Adds a new view to the blended image.

		The new view is divided in high and low frequencies with a mean blur kernel of size \f$\frac{1}{20}\f$ of the image size (see split_frequencies()).
		Each frequency is blended separately as a BlendedViewSimple.
		@param view View to add to the blended image.
		*/
		void blend(View const& view);

		void freeMemory(cv::Size imgSize);

	private:
		BlendedViewSimple m_low_freq;
		BlendedViewSimple m_high_freq;
	};
}

#endif
