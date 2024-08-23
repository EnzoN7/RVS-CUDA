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

#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include "Parameters.hpp"
#include "PoseTraces.hpp"
#include "JsonParser.hpp"

#include <string>
#include <vector>

#include <opencv2/core.hpp>

/**
@file Config.hpp
\brief The file containing the configuration
*/

namespace rvs
{
	namespace detail
	{
		/**\brief Working color space

		Doesn't need to be the same as the input our output color space
		*/
		enum class ColorSpace {
			YUV = 0,
			RGB = 1
		};
	}

	/**\brief View synthesis method

	For now only the triangle method is available
	*/
	namespace ViewSynthesisMethod
	{
		auto const triangles = "Triangles";
	}

	/**\brief Blending method

	\see BlendedView
	*/
	namespace BlendingMethod
	{
		auto const simple = "Simple";
		auto const multispectral = "Multispectral";
	}

	namespace detail
	{
		/**Precision*/
		extern float g_rescale;

		/**Working color space (RGB or YUV). Independent of the input or output formats*/
		extern ColorSpace g_color_space;
	}

	/** Enable OpenGL acceleration */
	extern bool g_with_opengl;

	/** Enable CUDA acceleration */
	extern bool g_with_cuda;

	/**
	\brief Configuration parameters
	*/
	class Config {
	public:
		/** Load configuration from file */
		static Config loadFromFile(std::string const& filename);

		/** Version of the configuration file */
		std::string version;

		/** Input camera names to lookup in the config file */
		std::vector<std::string> InputCameraNames;

		/** Virtual camera names to lookup in the config file */
		std::vector<std::string> VirtualCameraNames;

		/** input cameras parameters */
		std::vector<Parameters> params_real;

		/** virtual cameras parameters */
		std::vector<Parameters> params_virtual;

		/** filenames of the input color images */
		std::vector<std::string> texture_names;

		/** filenames of the input depth images */
		std::vector<std::string> depth_names;

		/** Name of the output files */
		std::vector<std::string> outfilenames;

		/** Name of the output masked files */
		std::vector<std::string> outmaskedfilenames;

		/** Name of the output masks */
		std::vector<std::string> outmaskfilenames;

		/**	Name of the output depth files	*/
		std::vector<std::string> outdepthfilenames;

		/**	Name of the output masked depth files	*/
		std::vector<std::string> outmaskdepthfilenames;

		/** Threshold for valid pixels */
		float validity_threshold = 5000.f;

		/** Method for view synthesis */
		std::string vs_method = "Triangles";

		/** Blending method (see BlendedView) */
		std::string blending_method = "Simple";

		/** Low frequency blending factor in BlendedViewMultiSpec */
		float blending_low_freq_factor;

		/** High frequency blending factor in BlendedViewMultiSpec */
		float blending_high_freq_factor;

		/** Blending factor in BlendedViewSimple */
		float blending_factor = 5.f;

		/** First frame to process (zero-based) */
		int start_frame = 0;

		/** Number of frames to process */
		int number_of_frames = 1;
		int number_of_output_frames = 1;

		/** The loaded pose trace */
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
		void setBlendingLowFreqFactor(json::Node root);
		void setBlendingHighFreqFactor(json::Node root);
		void setStartFrame(json::Node root);
		void setNumberOfFrames(json::Node root);
		void setNumberOfOutputFrames(json::Node root);

		static void setPrecision(json::Node root);
		static void setColorSpace(json::Node root);
	};
}

#endif