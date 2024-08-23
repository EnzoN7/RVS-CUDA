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

#include "EquirectangularUnprojector.hpp"

namespace rvs
{
	EquirectangularUnprojector::EquirectangularUnprojector(Parameters const& parameters)
		: Unprojector(parameters)
	{}

	cv::Mat3f EquirectangularUnprojector::unproject(cv::Mat2f image_pos, cv::Mat1f depth) const
	{
		auto world_pos = cv::Mat3f(image_pos.size());
		auto size = getParameters().getSize();
		auto hor_range = getParameters().getHorRange();
		auto ver_range = getParameters().getVerRange();

		auto const radperdeg = 0.01745329252f;
		auto phi0 = radperdeg * hor_range[1];
		auto theta0 = radperdeg * ver_range[1];
		auto dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;
		auto dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

		for (int i = 0; i != image_pos.rows; ++i) {
			for (int j = 0; j != image_pos.cols; ++j) {
				auto uv = image_pos(i, j);

				// Spherical coordinates
				auto phi = phi0 + dphi_du * uv[0];
				auto theta = theta0 + dtheta_dv * uv[1];

				// World position
				world_pos(i, j) = depth(i, j) * cv::Vec3f(
					std::cos(theta) * std::cos(phi),
					std::cos(theta) * std::sin(phi),
					std::sin(theta));
			}
		}

		return world_pos;
	}

	cv::Mat2f EquirectangularUnprojector::generateImagePos() const
	{
		auto image_pos = Unprojector::generateImagePos();
		float const eps = 1e-3f;

		for (int j = 0; j != image_pos.cols; ++j) {
			image_pos(0, j)[1] = eps;
			image_pos(image_pos.rows - 1, j)[1] = image_pos.rows - eps;
		}

		return image_pos;
	}
}
