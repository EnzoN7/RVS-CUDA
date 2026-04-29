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

#ifndef _SYNTHESIZED_VIEW_HPP_
#define _SYNTHESIZED_VIEW_HPP_

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "Config.hpp"
#include "View.hpp"
#include "Projector.hpp"
#include "SpaceTransformer.hpp"


/**
@file SynthesizedView.hpp
*/

namespace rvs
{
	/**
	 * \brief View computed with DIBR

	 The View obtained from one input reference view
	 * */
	class SynthesizedView : public View
	{
	public:
		/**
		\brief Default constructor
		*/
		SynthesizedView();

		/**
		\brief Destructor
		*/
		virtual ~SynthesizedView();

		/**
		\brief Set SpaceTransformer (input view to world)
		*/
		void setSpaceTransformer(SpaceTransformer const * object) { m_space_transformer = object; };

		/**
		\brief Compute this view from the input View

		The pipeline executes the following steps:
			- Unproject the pixels
			- Compute the transformation
			- Project the pixels locations in image coordinates
			- Rasterize the image with oversampling
		*/
		void compute(View& img);

	private:
		void computeBasicMode(View& img);

		void computeCudaMode(View& img);

	protected:
		/**
		\brief Rasterize the warped image, resulting in updates of color, depth and quality maps
		*/
		virtual void transform(cv::Mat3f input_color, cv::Mat2f input_positions, cv::Mat1f input_depth,
			cv::Size output_size, WrappingMethod wrapping_method) = 0;

		virtual void transformCudaMode(float3* input_color, float2* dev_scaled_uv, float* dev_input_depth, cv::Size input_size,
			cv::Size output_size, WrappingMethod wrapping_method) = 0;

		virtual void processTransformationCudaMode(Parameters inputParams, Parameters virtualParams, cv::Size input_size, View input, cv::Matx33f R, cv::Vec3f t,
			float2*& dev_virtual_uv, float*& dev_virtual_depth, WrappingMethod& wrapping_method) = 0;

	private:
		SpaceTransformer const *m_space_transformer = nullptr;
	};

	/**
	  \brief The algorithm to compute the view divides the input view in triangles, to avoid non disocclusion holes and get a first inpainting

	  This algorithm consists in warping the inputs using directly the disparity rather than performing back
	  and forth 3D re-projection of the pixels, which is sensitive to rounding operations. The translation
	  and rotation between the input and the target cameras are applied to get the result. The translation
	  is obtained by moving each pixel according to its disparity and the rotation is obtained with a homography.

	  For a camera movement in the image plane, for each pixel \f$p=(p_x,p_y)\f$, its translation \f$t=(t_x,t_y)\f$
	  in the image is given by the disparity \f$t=\frac{fT}{d}\f$ where \f$T=(T_x, T_y, T_z)\f$ is the translation
	  vector of the camera, d is the depth at \f$(p_x,p_y)\f$ and f is the focal length. For a movement along the Z
	  axis of the camera, it is \f$t'=\frac{T_z(p+t)}{f\left(d-T_z\right)}\f$

	  Three degrees of freedom of translation are obtained by combining the above equations. The rotation is obtained
	  with a simple homography. After the translation, each pixel p is displaced following the rotation of the camera
	  to get the final result \f$p'=(p_x',p_y',1)^T=\frac{1}{R_3\cdot(p_x,p_y,1)^T}R(p_x,p_y,1)^T\f$ where R is the
	  rotation matrix and $R_3$ its last row. With those equations, a map is obtained, indicating the new position
	  of each pixel in the new warped image.

	  The input view is divided into triangles with the pixels centers as vertices. The triangles are deformed using
	  the translation and rotation equations before being filled with interpolated colors. The colors are obtained
	  by the tri-linear interpolation between each three vertices of the triangle. Discontinuities between objects
	  creating disocclusions and tangential surfaces may lead to triangles with very elongated shapes, which will
	  not be kept in the final result, as they get eliminated during the blending phase.

	  In order to improve the final quality of the synthesized views, during the rasterization of the warped triangles,
	  the synthesized view is upscaled. Hence, the displacement of each pixel according to its depth is more precise
	  and the resulting image, once downscaled to the same size as the input, is sharper.

	  The quality of each pixel is given by the shape of the triangle it belongs to and the depth of the pixel (see BlendedView)
	  */
	class SynthetisedViewTriangle : public SynthesizedView {
	public:
		/**
		\brief Constructor
		*/
		SynthetisedViewTriangle();

	protected:
		/**
		\brief Rasterize the warped image, resulting in updates of color, depth and quality maps
		@param input_color Input color image
		@param input_positions New positions in the image of each pixel
		@param input_depth Input depth map
		@param output_size Size of the output image
		@param wrapping_method Output warping method (perpective or equirectangular, see Projector)
		\see transform_trianglesMethod()
		*/
		virtual void transform(cv::Mat3f input_color, cv::Mat2f input_positions, cv::Mat1f input_depth,
			cv::Size output_size, WrappingMethod wrapping_method);

		virtual void transformCudaMode(float3* input_color, float2* dev_scaled_uv, float* dev_input_depth, cv::Size input_size,
			cv::Size output_size, WrappingMethod wrapping_method);

		virtual void processTransformationCudaMode(Parameters inputParams, Parameters virtualParams, cv::Size input_size, View input, cv::Matx33f R, cv::Vec3f t,
			float2*& dev_virtual_uv, float*& dev_virtual_depth, WrappingMethod& wrapping_method);
	};
}

#endif