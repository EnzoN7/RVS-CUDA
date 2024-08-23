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

#ifndef _SPACE_TRANSFORMER_HPP_
#define _SPACE_TRANSFORMER_HPP_

#include "Projector.hpp"
#include "Unprojector.hpp"
#include "Config.hpp"

#if WITH_OPENGL
#include "Shader.hpp"
#endif

#include "opencv2/imgproc.hpp"

#include <memory>

namespace rvs
{
	class SpaceTransformer
	{
	public:
		SpaceTransformer();
		virtual ~SpaceTransformer();

		Parameters const& getInputParameters() const;
		Parameters const& getVirtualParameters() const;
		cv::Vec3f get_translation() const;
		cv::Matx33f get_rotation() const;

		virtual void set_targetPosition(Parameters const *params_virtual);
		virtual void set_inputPosition(Parameters const *params_real);

		float get_sensor_size() const { return static_cast<float>(m_input_parameters->getSize().width); };
		cv::Size get_size() const { return m_output_parameters->getSize(); };
	private:
		Parameters const *m_input_parameters;
		Parameters const *m_output_parameters;
	};

	class GenericTransformer : public SpaceTransformer
	{
	public:
		cv::Mat2f project(cv::Mat3f world_pos, /*out*/ cv::Mat1f& depth, /*out*/ WrappingMethod& wrapping_method) const;
		cv::Mat3f unproject(cv::Mat2f image_pos, cv::Mat1f depth) const;

		void set_targetPosition(Parameters const *params_virtual) override;
		void set_inputPosition(Parameters const *params_real) override;

		/** Generate input image positions */
		cv::Mat2f generateImagePos() const;

	protected:
		std::unique_ptr<Unprojector> m_unprojector;
		std::unique_ptr<Projector> m_projector;
	};

	class OpenGLTransformer : public SpaceTransformer {};
}

#endif
