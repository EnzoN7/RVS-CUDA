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

#include "PoseTraces.hpp"

#include <fstream>
#include <regex>

namespace rvs
{
	PoseTrace PoseTrace::loadFrom(std::istream & stream)
	{
		std::string line;
		std::getline(stream, line);

		std::regex re_header("\\s*X\\s*,\\s*Y\\s*,\\s*Z\\s*,\\s*Yaw\\s*,\\s*Pitch\\s*,\\s*Roll\\s*");
		if (!std::regex_match(line, re_header)) {
			throw std::runtime_error("Format error in the pose trace header");
		}

		PoseTrace trace;
		std::regex re_row("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)");
		std::regex re_empty("\\s*");
		bool trailing_empty_lines = false;

		while (std::getline(stream, line)) {
			std::smatch match;
			if (!trailing_empty_lines && std::regex_match(line, match, re_row)) {
				trace.push_back({
					cv::Vec3f(
						std::stof(match[1].str()),
						std::stof(match[2].str()),
						std::stof(match[3].str())),
					cv::Vec3f(
						std::stof(match[4].str()),
						std::stof(match[5].str()),
						std::stof(match[6].str()))
					});
			}
			else if (std::regex_match(line, re_empty)) {
				trailing_empty_lines = true;
			}
			else {
				throw std::runtime_error("Format error in a pose trace row");
			}
		}

		return trace;
	}

	PoseTrace PoseTrace::loadFromFile(std::string const& filename)
	{
		std::ifstream stream(filename);
		if (!stream.good()) {
			throw std::runtime_error("Failed to load pose trace");
		}
		//TODO if pose trace if derivative of position
		/*PoseTrace derivated = loadFrom(stream);
		for (int i = 1; i < derivated.size(); ++i)
		{
			derivated[i].position += derivated[i - 1].position;
		}
		return derivated;*/
		//TODO if pose trace is position
		return loadFrom(stream);
	}
}