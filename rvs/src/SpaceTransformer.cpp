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

#include "SpaceTransformer.hpp"
#include "EquirectangularProjector.hpp"
#include "EquirectangularUnprojector.hpp"
#include "PerspectiveProjector.hpp"
#include "PerspectiveUnprojector.hpp"

#include <cassert>

namespace rvs
{
	SpaceTransformer::SpaceTransformer()
		: m_input_parameters(nullptr)
		, m_output_parameters(nullptr)
	{
	}

	SpaceTransformer::~SpaceTransformer() {}

	Parameters const& SpaceTransformer::getInputParameters() const
	{
		assert(m_input_parameters);
		return *m_input_parameters;
	}

	Parameters const& SpaceTransformer::getVirtualParameters() const
	{
		assert(m_output_parameters);
		return *m_output_parameters;
	}

	cv::Vec3f SpaceTransformer::get_translation() const
	{
		auto t_from = getInputParameters().getPosition();
		auto R_to = getVirtualParameters().getRotationMatrix();
		auto t_to = getVirtualParameters().getPosition();

		return -R_to.t()*(t_to - t_from);
	}

	cv::Matx33f SpaceTransformer::get_rotation() const
	{
		auto R_from = getInputParameters().getRotationMatrix();
		auto R_to = getVirtualParameters().getRotationMatrix();

		return R_to.t()*R_from;
	}
	void SpaceTransformer::set_inputPosition(Parameters const *parameters)
	{
		m_input_parameters = parameters;
	}

	void SpaceTransformer::set_targetPosition(Parameters const *parameters)
	{
		m_output_parameters = parameters;
	}

	cv::Mat2f GenericTransformer::project(cv::Mat3f world_pos, /*out*/ cv::Mat1f& depth, /*out*/ WrappingMethod& wrapping_method) const
	{
		return m_projector->project(world_pos, depth, wrapping_method);
	}
	cv::Mat3f GenericTransformer::unproject(cv::Mat2f image_pos, cv::Mat1f depth) const
	{
		return m_unprojector->unproject(image_pos, depth);
	}

	void GenericTransformer::set_inputPosition(Parameters const *parameters)
	{
		assert(parameters);
		SpaceTransformer::set_inputPosition(parameters);

		if (parameters->getProjectionType() == ProjectionType::equirectangular) {
			m_unprojector.reset(new EquirectangularUnprojector(*parameters));
		}
		else if (parameters->getProjectionType() == ProjectionType::perspective) {
			m_unprojector.reset(new PerspectiveUnprojector(*parameters));
		}
		else {
			std::ostringstream what;
			what << "Unknown projection type \"" << parameters->getProjectionType() << "\"";
			throw std::runtime_error(what.str());
		}
	}

	void GenericTransformer::set_targetPosition(Parameters const *parameters)
	{
		assert(parameters);
		SpaceTransformer::set_targetPosition(parameters);

		if (parameters->getProjectionType() == ProjectionType::equirectangular) {
			m_projector.reset(new EquirectangularProjector(*parameters));
		}
		else if (parameters->getProjectionType() == ProjectionType::perspective) {
			m_projector.reset(new PerspectiveProjector(*parameters));
		}
		else {
			std::ostringstream what;
			what << "Unknown projection type \"" << parameters->getProjectionType() << "\"";
			throw std::runtime_error(what.str());
		}
	}

	cv::Mat2f GenericTransformer::generateImagePos() const
	{
		return m_unprojector->generateImagePos();
	}
}