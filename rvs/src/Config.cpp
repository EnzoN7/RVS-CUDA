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

#include "Config.hpp"
#include "JsonParser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <map>


namespace rvs
{
	bool g_verbose = false;

	namespace detail
	{
		float const defaultPrecision = 1.f;
		ColorSpace const defaultColorSpace = ColorSpace::YUV;

		float g_rescale = defaultPrecision;
		ColorSpace g_color_space = defaultColorSpace;
	}

	bool g_with_opengl = false;
	bool g_with_cuda = false;

	Config Config::loadFromFile(std::string const& filename)
	{
		if(g_verbose)
			std::cout << '\n';
		std::ifstream stream(filename);
		if (!stream.good()) {
			std::ostringstream what;
			what << "Failed to open configuration file \"" << filename << "\" for reading";
			throw std::runtime_error(what.str());
		}
		auto root = json::Node::readFrom(stream);

		Config config;

		config.setVersionFrom(root);
		config.setInputCameraNamesFrom(root);
		config.setVirtualCameraNamesFrom(root);
		config.setInputCameraParameters(root);
		config.setVirtualCameraParameters(root);
		config.setInputFilepaths(root, "ViewImageNames", config.texture_names);
		config.setInputFilepaths(root, "DepthMapNames", config.depth_names);
		config.setOutputFilepaths(root, "OutputFiles", config.outfilenames);
		config.setOutputFilepaths(root, "MaskedOutputFiles", config.outmaskedfilenames);
		config.setOutputFilepaths(root, "OutputMasks", config.outmaskfilenames);
		config.setOutputFilepaths(root, "DepthOutputFiles", config.outdepthfilenames);
		config.setOutputFilepaths(root, "MaskedDepthOutputFiles", config.outmaskdepthfilenames);
		config.setValidityThreshold(root);
		config.setSynthesisMethod(root);
		config.setBlendingMethod(root);
		config.setBlendingFactor(root);
		config.setBlendingLowFreqFactor(root);
		config.setBlendingHighFreqFactor(root);
		config.setStartFrame(root);
		config.setNumberOfFrames(root);
		config.setNumberOfOutputFrames(root);

		setPrecision(root);
		setColorSpace(root);

		auto node = root.optional("VirtualPoseTraceName");
		if (node) {
			auto filepath = node.asString();
			if(g_verbose)
				std::cout << "VirtualPoseTraceName: " << filepath << '\n';
			config.loadPoseTraceFromFile(filepath);
		}

		return config;
	}

	std::vector<Parameters> Config::loadCamerasParametersFromFile(std::string const& filepath, std::vector<std::string> names, json::Node overrides)
	{
		// Load the camera parameters
		std::ifstream stream(filepath);
		if (!stream.good()) {
			std::ostringstream what;
			what << "Failed to open camera parameters file \"" << filepath << "\" for reading";
			throw std::runtime_error(what.str());
		}
		auto root = json::Node::readFrom(stream);
		auto version_ = root.require("Version").asString();
		if (version_.substr(0, 2) != "2." &&
			version_.substr(0, 2) != "3." &&
			version_.substr(0, 2) != "4.") {
			throw std::runtime_error("Version of the camera parameters file is not compatible with this version of RVS");
		}

		// Load parameters (with overrides) and index by camera name
		std::map<std::string, Parameters> index;
		auto cameras = root.require("cameras");
		for (auto i = 0u; i != cameras.size(); ++i) {
			auto node = cameras.at(i);
			auto name = node.require("Name").asString();
			if (index.count(name)) {
				std::ostringstream what;
				what << "Camera parameters file has duplicate camera '" << name << "'";
				throw std::runtime_error(what.str());
			}
			if (overrides) {
				node.setOverrides(overrides);
			}
			index.emplace(name, Parameters::readFrom(node));
		}

		// Assign in the requested order
		std::vector<Parameters> parameters;
		for (auto name : names) {
			try {
				if (g_verbose) 
				{
					std::cout << "  * " << name << ": ";
					index.at(name).printTo(std::cout);
					std::cout << '\n';
				}
				parameters.push_back(index.at(name));
			}
			catch (std::out_of_range&) {
				std::ostringstream what;
				what << "Camera parameters file does not have camera '" << name << "'";
				throw std::runtime_error(what.str());
			}
		}
		return parameters;
	}

	void Config::loadPoseTraceFromFile(std::string const& filepath)
	{
		if (!filepath.empty()) {
			pose_trace = PoseTrace::loadFromFile(filepath);
			if (static_cast<unsigned>(start_frame + number_of_frames) > pose_trace.size()) {
				throw std::runtime_error("Error: Number of frames to process is larger then number of entries in pose trace file");
			}
			if (g_verbose)
				std::cout << std::endl << "Using pose trace with " << pose_trace.size() << " entries" << std::endl;
		}
	}

	void Config::setVersionFrom(json::Node root)
	{
		version = root.require("Version").asString();
		if (version.substr(0, 2) != "2." &&
			version.substr(0, 2) != "3.") {
			throw std::runtime_error("Configuration file does not match the RVS version");
		}
		if (g_verbose)
			std::cout << "Version: " << version << '\n';
	}

	void Config::setInputCameraNamesFrom(json::Node root)
	{
		auto node = root.require("InputCameraNames");
		for (auto i = 0u; i != node.size(); ++i) {
			InputCameraNames.push_back(node.at(i).asString());
		}
		if (g_verbose)
		{
			std::cout << "InputCameraNames:";
			for (auto x : InputCameraNames) {
				std::cout << ' ' << x;
			}
			std::cout << '\n';
		}
	}

	void Config::setVirtualCameraNamesFrom(json::Node root)
	{
		auto node = root.require("VirtualCameraNames");
		for (auto i = 0u; i != node.size(); ++i) {
			VirtualCameraNames.push_back(node.at(i).asString());
		}
		if (g_verbose)
		{
			std::cout << "VirtualCameraNames:";
			for (auto x : VirtualCameraNames) {
				std::cout << ' ' << x;
			}
			std::cout << '\n';
		}
	}

	void Config::setInputCameraParameters(json::Node root)
	{
		auto filepath = root.require("InputCameraParameterFile").asString();
		if (g_verbose)
			std::cout << "InputCameraParameterFile: " << filepath << '\n';
		auto overrides = root.optional("InputOverrides");
		if (overrides) {
			if (g_verbose)
				std::cout << "InputOverrides: " << overrides.size() << " keys\n";
		}
		params_real = loadCamerasParametersFromFile(filepath, InputCameraNames, overrides);
	}

	void Config::setVirtualCameraParameters(json::Node root)
	{
		auto filepath = root.require("VirtualCameraParameterFile").asString();
		if (g_verbose)
			std::cout << "VirtualCameraParameterFile: " << filepath << '\n';
		auto overrides = root.optional("VirtualOverrides");
		if (overrides) {
			if (g_verbose)
				std::cout << "VirtualOverrides: " << overrides.size() << " keys\n";
		}
		params_virtual = loadCamerasParametersFromFile(filepath, VirtualCameraNames, overrides);
	}

	void Config::setInputFilepaths(json::Node root, char const *name, std::vector<std::string>& filepaths)
	{
		auto node = root.require(name);
		if (node.size() != InputCameraNames.size()) {
			std::ostringstream what;
			what << "Length of " << name << " should match with InputCameraNames";
			throw std::runtime_error(what.str());
		}
		for (auto i = 0u; i != node.size(); ++i) {
			filepaths.push_back(node.at(i).asString());
		}
		if (g_verbose)
		{
			std::cout << name << ':';
			for (auto x : filepaths) {
				std::cout << "\n  * " << x;
			}
			std::cout << '\n';
		}
	}

	void Config::setOutputFilepaths(json::Node root, char const *name, std::vector<std::string>& filepaths)
	{
		auto node = root.optional(name);
		if (node) {
			if (node.size() != VirtualCameraNames.size()) {
				std::ostringstream what;
				what << "Length of " << name << " should match with VirtualCameraNames";
				throw std::runtime_error(what.str());
			}
			for (auto i = 0u; i != node.size(); ++i) {
				filepaths.push_back(node.at(i).asString());
			}
			if (g_verbose)
			{
				std::cout << name << ':';
				for (auto x : filepaths) {
					std::cout << "\n  * " << x;
				}
				std::cout << '\n';
			}
		}
	}

	void Config::setValidityThreshold(json::Node root)
	{
		auto node = root.optional("ValidityThreshold");
		if (node) {
			validity_threshold = static_cast<float>(node.asDouble());
			if (g_verbose)
				std::cout << "ValidityThreshold: " << validity_threshold << '\n';
		}
	}

	void Config::setSynthesisMethod(json::Node root)
	{
		auto node = root.optional("ViewSynthesisMethod");
		if (node) {
			vs_method = node.asString();
		}
	}

	void Config::setBlendingMethod(json::Node root)
	{
		auto node = root.optional("BlendingMethod");
		if (node) {
			blending_method = node.asString();
			if (g_verbose)
				std::cout << "BlendingMethod: " << blending_method << '\n';
		}
	}

	void Config::setBlendingFactor(json::Node root)
	{
		auto node = root.optional("BlendingFactor");
		if (node) {
			blending_factor = static_cast<float>(node.asDouble());
			if (g_verbose)
				std::cout << "BlendingFactor: " << blending_factor << '\n';
		}
	}

	void Config::setBlendingLowFreqFactor(json::Node root)
	{
		if (blending_method == BlendingMethod::multispectral) {
			blending_low_freq_factor = static_cast<float>(root.require("BlendingLowFreqFactor").asDouble());
			if (g_verbose)
				std::cout << "BlendingLowFreqFactor: " << blending_low_freq_factor << '\n';
		}
	}

	void Config::setBlendingHighFreqFactor(json::Node root)
	{
		if (blending_method == BlendingMethod::multispectral) {
			blending_high_freq_factor = static_cast<float>(root.require("BlendingHighFreqFactor").asDouble());
			if (g_verbose)
				std::cout << "BlendingHighFreqFactor: " << blending_high_freq_factor << '\n';
		}
	}

	void Config::setStartFrame(json::Node root)
	{
		auto node = root.optional("StartFrame");
		if (node) {
			start_frame = node.asInt();
			if (g_verbose)
				std::cout << "StartFrame: " << start_frame << '\n';
		}
	}

	void Config::setNumberOfFrames(json::Node root)
	{
		auto node = root.optional("NumberOfFrames");
		if (node) {
			number_of_frames = node.asInt();
			if (g_verbose)
				std::cout << "NumberOfFrames: " << number_of_frames << '\n';
		}
	}
	void Config::setNumberOfOutputFrames(json::Node root)
	{
		auto node = root.optional("NumberOfOutputFrames");
		if (node) {
			number_of_output_frames = node.asInt();
			if (g_verbose)
				std::cout << "NumberOfOutputFrames: " << number_of_output_frames << '\n';
		}
		else 
			number_of_output_frames = number_of_frames;
	}

	void Config::setPrecision(json::Node root)
	{
		auto node = root.optional("Precision");
		if (node) {
			detail::g_rescale = static_cast<float>(node.asDouble());
			if (g_verbose)
				std::cout << "Precision: " << detail::g_rescale << '\n';
		}
		else {
			detail::g_rescale = detail::defaultPrecision;
		}
	}

	void Config::setColorSpace(json::Node root)
	{
		auto node = root.optional("ColorSpace");
		if (node) {
			if (node.asString() == "YUV") {
				detail::g_color_space = detail::ColorSpace::YUV;
				if (g_verbose)
					std::cout << "ColorSpace: YUV\n";
			}
			else if (node.asString() == "RGB") {
				detail::g_color_space = detail::ColorSpace::RGB;
				if (g_verbose)
					std::cout << "ColorSpace: RGB\n";
			}
			else {
				throw std::runtime_error("Unknown color space");
			}
		}
		else {
			detail::g_color_space = detail::defaultColorSpace;
		}
	}
}
