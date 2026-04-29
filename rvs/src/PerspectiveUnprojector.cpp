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

#include "PerspectiveUnprojector.hpp"

#include <limits>
#include <iostream>
auto const NaN = std::numeric_limits<float>::quiet_NaN();

namespace rvs
{
	PerspectiveUnprojector::PerspectiveUnprojector(Parameters const& parameters)
		: Unprojector(parameters)
	{}

	cv::Mat3f PerspectiveUnprojector::unproject(cv::Mat2f image_pos, cv::Mat1f depth) const
	{
		assert(image_pos.size() == depth.size());

		auto f = getParameters().getFocal();
		auto p = getParameters().getPrinciplePoint();

		cv::Mat3f world_pos(image_pos.size(), cv::Vec3f::all(NaN));

		for (int i = 0; i != image_pos.rows; ++i) {
			for (int j = 0; j != image_pos.cols; ++j) {
				auto uv = image_pos(i, j);
				auto d = depth(i, j);

				// OMAF Referential: x forward, y left, z up
				// Image plane: x right, y down

				if (d > 0.f) {
					world_pos(i, j) = cv::Vec3f(
						d,
						-(d / f[0]) * (uv[0] - p[0]),
						-(d / f[1]) * (uv[1] - p[1]));
				}
			}
		}

		return world_pos;
	}
}