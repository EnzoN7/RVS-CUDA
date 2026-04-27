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

#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include "Parameters.hpp"
#include "PoseTraces.hpp"
#include "JsonParser.hpp"

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>


namespace rvs
{
	namespace detail
	{
		enum class ColorSpace
		{
			YUV = 0,
			RGB = 1
		};
	}

	namespace ViewSynthesisMethod
	{
		auto const triangles = "Triangles";
	}

	namespace BlendingMethod
	{
		auto const simple = "Simple";
	}

	namespace detail
	{
		extern float g_rescale;
		extern bool g_FP32;
		extern bool g_FP64;
		extern ColorSpace g_color_space;
		extern int numFramesSeq;
	}

	class Config {
	public:
		static Config loadFromFile(std::string const& filename);

		std::string version;
		std::vector<std::string> InputCameraNames;
		std::vector<std::string> VirtualCameraNames;
		std::vector<Parameters> params_real;
		std::vector<Parameters> params_virtual;
		std::vector<std::string> texture_names;
		std::vector<std::string> depth_names;
		std::vector<std::string> outfilenames;
		std::vector<std::string> outmaskedfilenames;
		std::vector<std::string> outmaskfilenames;
		std::vector<std::string> outdepthfilenames;
		std::vector<std::string> outmaskdepthfilenames;
		float validity_threshold = 5000.f;
		std::string vs_method = "Triangles";
		std::string blending_method = "Simple";
		float blending_factor = 5.f;
		int start_frame = 0;
		int number_of_frames = 1;
		int number_of_output_frames = 1;
		PoseTrace pose_trace;

	private:
		Config() = default;

		std::vector<Parameters> loadCamerasParametersFromFile(std::string const& filepath, std::vector<std::string> names, json::Node overrides);
		void loadPoseTraceFromFile(std::string const& filepath);

		void setVersionFrom(json::Node root);
		void setInputCameraNamesFrom(json::Node root);
		void setVirtualCameraNamesFrom(json::Node root);
		void setInputCameraParameters(json::Node root);
		void setVirtualCameraParameters(json::Node root);
		void setInputFilepaths(json::Node root, char const *name, std::vector<std::string>&);
		void setOutputFilepaths(json::Node root, char const *name, std::vector<std::string>&);
		void setValidityThreshold(json::Node root);
		void setSynthesisMethod(json::Node root);
		void setBlendingMethod(json::Node root);
		void setBlendingFactor(json::Node root);
		void setStartFrame(json::Node root);
		void setNumberOfFrames(json::Node root);
		void setNumberOfOutputFrames(json::Node root);

		static void setPrecision(json::Node root);
		static void setColorSpace(json::Node root);
	};
}

#endif