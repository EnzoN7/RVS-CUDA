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

#ifndef _SYNTHESIZED_VIEW_HPP_
#define _SYNTHESIZED_VIEW_HPP_

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

#include "Config.hpp"
#include "View.hpp"
#include "Projector.hpp"
#include "SpaceTransformer.hpp"
#include "transform_triangle.cuh"
#include "unproject_project.cuh"
#include "scale_uv.cuh"

namespace rvs
{
	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	class SynthesizedView : public View
	{
	  public:
		SynthesizedView();
		virtual ~SynthesizedView();

		virtual void setSpaceTransformer(SpaceTransformer const* object) = 0;
		virtual void compute(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2) = 0;
		virtual void waitAndGetVirtualComponents(color_t*& devVirtualColor, channel_t*& devVirtualDepth, channel_t*& devVirtualValidity) = 0;

	  protected:
		virtual void synthesizeImage(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2) = 0;

		virtual void unprojectTo3D_projectTo2D(Parameters& virtualParams, cudaStream_t& stream) = 0;

	  private:
		SpaceTransformer const* m_space_transformer = nullptr;
	};

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	class SynthetizedViewTriangle : public SynthesizedView<position_t, color_t, channel_t>
	{
	  public:
		SynthetizedViewTriangle(cv::Size realSize, cv::Size virtualSize, SpaceTransformer const* object);
		~SynthetizedViewTriangle();

		void compute(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2);

		void setSpaceTransformer(SpaceTransformer const* object)
		{
			m_space_transformer = object;
		};

		void waitAndGetVirtualComponents(color_t*& devVirtualColor, channel_t*& devVirtualDepth, channel_t*& devVirtualValidity)
		{
			cudaEventSynchronize(m_synthesizeVirtualDepthValidity);

			devVirtualColor = m_devVirtualColor;
			devVirtualDepth = m_devVirtualDepth;
			devVirtualValidity = m_devVirtualValidity;
		}

	  protected:
		void synthesizeImage(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2);

		void unprojectTo3D_projectTo2D(Parameters& virtualParams, cudaStream_t& stream);

	  private:
		SpaceTransformer const* m_space_transformer;
		Parameters m_inputParams;

		cv::Size m_realSize;
		cv::Size m_virtualSize;
		encoded_t<channel_t>* m_devOutputDepthValidity;
		int* m_devColorLock;
		int* m_devOutputTriId;

		color_t* m_devNormalizedColor;
		channel_t* m_devNormalizedDepth;

		position_t* m_devTransformedPosition;
		channel_t* m_devTransformedDepth;

		color_t* m_devVirtualColor;
		channel_t* m_devVirtualDepth;
		channel_t* m_devVirtualValidity;

		cudaEvent_t m_synthesizeVirtualDepthValidity;
		cudaEvent_t m_initDepthValidity;
		cudaEvent_t m_colorizeTriangles;
		cudaEvent_t m_projection;

		PrecomputedParams m_precomputedParams;
		cv::Size m_lastSize;
		cv::Vec2f m_lastHorRange;
		cv::Vec2f m_lastVerRange;
		cv::Vec2f m_lastF;
		cv::Vec2f m_lastP;

		std::future<void> m_futureProjection;

		WrappingMethod m_wrappingMethod;

		void prepareParameters(cv::Size size, cv::Vec2f hor_range, cv::Vec2f ver_range,
							   cv::Matx33f R, cv::Vec3f t, cv::Vec2f f, cv::Vec2f p);
	};
}

#endif