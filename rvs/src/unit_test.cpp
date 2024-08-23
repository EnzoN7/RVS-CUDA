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

#include "yaffut.hpp"

#include "PerspectiveProjector.hpp"
#include "PerspectiveUnprojector.hpp"
#include "EquirectangularProjector.hpp"
#include "EquirectangularUnprojector.hpp"
#include "PoseTraces.hpp"
#include "JsonParser.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;

namespace testing
{
	namespace erp
	{
		rvs::Parameters generateParameters()
		{
			std::istringstream text(R"({
				"Name": "v0",
				"Projection": "Equirectangular",
				"Position": [0, 0, 0],
				"Rotation": [0, 0, 0],
				"Depthmap": 1,
				"Background": 0,
				"Depth_range": [0, 1],
				"Resolution": [5, 5],
				"BitDepthColor": 13,
				"BitDepthDepth": 15,
				"ColorSpace": "YUV420",
				"DepthColorSpace": "YUV420",
				"Hor_range": [-180, 180],
				"Ver_range": [-90, 90]
		})");

			// Construct unprojector
			auto root = json::Node::readFrom(text);
			return rvs::Parameters::readFrom(root);
		}

		cv::Mat2f generateReferenceImagePos()
		{
			auto referenceImagePos = cv::Mat2f(5, 5);
			for (int j = 0; j != 5; ++j) {
				referenceImagePos(0, j) = cv::Vec2f(j + 0.5f, 0.001f);
				referenceImagePos(1, j) = cv::Vec2f(j + 0.5f, 1.500f);
				referenceImagePos(2, j) = cv::Vec2f(j + 0.5f, 2.500f);
				referenceImagePos(3, j) = cv::Vec2f(j + 0.5f, 3.500f);
				referenceImagePos(4, j) = cv::Vec2f(j + 0.5f, 4.999f);
			}
			return referenceImagePos;
		}

		cv::Mat1f generateReferenceDepth()
		{
			cv::Mat1f referenceDepth = cv::Mat1f::ones(5, 5);
			referenceDepth(2, 2) = 2.f;
			return referenceDepth;
		}

		cv::Mat3f generateReferenceWorldPos(cv::Mat2f imagePos, cv::Mat1f depth)
		{
			auto referenceWorldPos = cv::Mat3f(5, 5);
			for (int i = 0; i != 5; ++i) {
				for (int j = 0; j != 5; ++j) {
					auto phi = CV_PI - CV_2PI * (imagePos(i, j)[0] / 5.);
					auto theta = CV_PI / 2 - CV_PI * (imagePos(i, j)[1] / 5.);
					referenceWorldPos(i, j)[0] = static_cast<float>(depth(i, j) * std::cos(theta) * std::cos(phi));
					referenceWorldPos(i, j)[1] = static_cast<float>(depth(i, j) * std::cos(theta) * std::sin(phi));
					referenceWorldPos(i, j)[2] = static_cast<float>(depth(i, j) * std::sin(theta));
				}
			}
			return referenceWorldPos;
		}
	}
}

namespace rvs
{
	extern bool g_verbose;
}

FUNC(Test_EquirectangularUnprojector_generateImagePos)
{
	// Construct unprojector
	auto parameters = testing::erp::generateParameters();
	rvs::EquirectangularUnprojector unprojector(parameters);

	// Generate image positions
	auto actualImagePos = unprojector.generateImagePos();

	// Check image positions against reference
	auto referenceImagePos = testing::erp::generateReferenceImagePos();
	auto generatedImagePosError = cv::norm(referenceImagePos, actualImagePos, cv::NORM_INF);
	CHECK(generatedImagePosError < 1e-6f);
}

FUNC(Test_EquirectangularUnprojector_unproject)
{
	// Construct unprojector
	auto parameters = testing::erp::generateParameters();
	rvs::EquirectangularUnprojector unprojector(parameters);

	// Unproject to world coordinates
	auto imagePos = testing::erp::generateReferenceImagePos();
	auto depth = testing::erp::generateReferenceDepth();
	auto actualWorldPos = unprojector.unproject(imagePos, depth);

	// Check world positions against reference
	auto referenceWorldPos = testing::erp::generateReferenceWorldPos(imagePos, depth);
	auto actualWorldPosError = cv::norm(referenceWorldPos, actualWorldPos, cv::NORM_INF);
	CHECK(actualWorldPosError < 1e-6f);
}

FUNC(Test_EquirectangularProjector_project)
{
	// Construct projector
	auto parameters = testing::erp::generateParameters();
	rvs::EquirectangularProjector projector(parameters);

	// Project to image coordinates
	auto referenceImagePos = testing::erp::generateReferenceImagePos();
	auto referenceDepth = testing::erp::generateReferenceDepth();
	auto worldPos = testing::erp::generateReferenceWorldPos(referenceImagePos, referenceDepth);
	cv::Mat1f actualDepth;
	rvs::WrappingMethod actualWrappingMethod = rvs::WrappingMethod::none;
	auto actualImagePos = projector.project(worldPos, actualDepth, actualWrappingMethod);

	// Check image positions against reference
	auto referenceImagePosError = cv::norm(referenceImagePos, actualImagePos, cv::NORM_INF);
	CHECK(referenceImagePosError < 1e-4f);

	// Check depth map against reference
	auto referenceDepthError = cv::norm(referenceDepth, actualDepth, cv::NORM_INF);
	CHECK(referenceDepthError < 1e-6f);
	
	// Check wrapping
	CHECK(actualWrappingMethod == rvs::WrappingMethod::horizontal);
}

namespace testing
{
	namespace persp
	{
		rvs::Parameters generateParameters()
		{
			std::istringstream text(R"({
			"Name"				: "v0",
			"Projection"		: "Perspective",
			"Position"			: [5, 3, 9],
			"Rotation"			: [10, 5, -3],
			"Depthmap"			: 1,
			"Background"		: 0,
			"Depth_range"		: [0, 1],
			"Resolution"		: [4, 3],
			"Focal"				: [3, 3],
			"Principle_point"	: [2, 1],
			"BitDepthColor"		: 10,
			"BitDepthDepth"		: 10,
			"ColorSpace"		: "YUV420",
			"DepthColorSpace"	: "YUV420"
		})");

			auto root = json::Node::readFrom(text);
			return rvs::Parameters::readFrom(root);
		}

		cv::Mat2f generateReferenceImagePos()
		{
			auto referenceImagePos = cv::Mat2f(3, 4);
			for (int i = 0; i != 3; ++i) {
				for (int j = 0; j != 4; ++j) {
					referenceImagePos(i, j) = cv::Vec2f(j + 0.5f, i + 0.5f);
				}
			}
			return referenceImagePos;
		}

		cv::Mat1f generateReferenceDepth()
		{
			cv::Mat1f referenceDepth = cv::Mat1f::ones(3, 4);
			referenceDepth(2, 2) = 2.f;
			return referenceDepth;
		}

		cv::Mat3f generateReferenceWorldPos(cv::Mat2f imagePos, cv::Mat1f depth)
		{
			auto referenceWorldPos = cv::Mat3f(3, 4);
			for (int i = 0; i != 3; ++i) {
				for (int j = 0; j != 4; ++j) {
					auto sensorPos = cv::Vec3d(
						3.,
						2. - imagePos(i, j)[0],
						1. - imagePos(i, j)[1]);
					referenceWorldPos(i, j) = cv::Vec3f(sensorPos * (depth(i, j) / 3.));
				}
			}
			return referenceWorldPos;
		}
	}
}

FUNC(Test_PerspectiveUnprojector_generateImagePos)
{
	// Construct unprojector
	auto parameters = testing::persp::generateParameters();
	rvs::PerspectiveUnprojector unprojector(parameters);

	// Generate image positions
	auto actualImagePos = unprojector.generateImagePos();

	// Check image positions against reference
	auto referenceImagePos = testing::persp::generateReferenceImagePos();
	auto generatedImagePosError = cv::norm(referenceImagePos, actualImagePos, cv::NORM_INF);
	CHECK(generatedImagePosError < 1e-6f);
}

FUNC(Test_PerspectiveUnprojector_unproject)
{
	// Construct unprojector
	auto parameters = testing::persp::generateParameters();
	rvs::PerspectiveUnprojector unprojector(parameters);

	// Unproject to world coordinates
	auto imagePos = testing::persp::generateReferenceImagePos();
	auto depth = testing::persp::generateReferenceDepth();
	auto actualWorldPos = unprojector.unproject(imagePos, depth);

	// Check world positions against reference
	auto referenceWorldPos = testing::persp::generateReferenceWorldPos(imagePos, depth);
	auto actualWorldPosError = cv::norm(referenceWorldPos, actualWorldPos, cv::NORM_INF);
	CHECK(actualWorldPosError < 1e-6f);
}

FUNC(Test_PerspectiveProjector_project)
{
	// Construct projector
	auto parameters = testing::persp::generateParameters();
	rvs::PerspectiveProjector projector(parameters);

	// Project to image coordinates
	auto referenceImagePos = testing::persp::generateReferenceImagePos();
	auto referenceDepth = testing::persp::generateReferenceDepth();
	auto worldPos = testing::persp::generateReferenceWorldPos(referenceImagePos, referenceDepth);
	cv::Mat1f actualDepth;
	rvs::WrappingMethod actualWrappingMethod = rvs::WrappingMethod::horizontal;
	auto actualImagePos = projector.project(worldPos, actualDepth, actualWrappingMethod);

	// Check image positions against reference
	auto referenceImagePosError = cv::norm(referenceImagePos, actualImagePos, cv::NORM_INF);
	CHECK(referenceImagePosError < 1e-6f);

	// Check depth map against reference
	auto referenceDepthError = cv::norm(referenceDepth, actualDepth, cv::NORM_INF);
	CHECK(referenceDepthError < 1e-6f);

	// Check wrapping
	CHECK(actualWrappingMethod == rvs::WrappingMethod::none);
}

FUNC(Test_JsonParser_readFrom)
{
	std::istringstream stream(R"(
{
	"Content_name": "ClassroomVideo",
	"BoundingBox_center" : [0.0, 42.0, -1e3],
	"Fps" : 30,
	"cameras" :
	[
		{
			"Name": "v0",
			"Background" : 0
		},
		{
		}
	]
})");
	
	auto root = json::Node::readFrom(stream);
	EQUAL(root.optional("Content_name").asString(), "ClassroomVideo");	
	auto center = root.optional("BoundingBox_center");
	EQUAL(center.size(), 3u);
	EQUAL(center.at(0).asDouble(), 0.0);
	EQUAL(center.at(1).asDouble(), 42.0);
	EQUAL(center.at(2).asDouble(), -1e3);
	EQUAL(root.optional("Fps").asDouble(), 30.0);
	auto cameras = root.optional("cameras");
	EQUAL(cameras.size(), 2u);
	auto cam0 = cameras.at(0);
	EQUAL(cam0.optional("Name").asString(), "v0");
	EQUAL(cam0.optional("Background").asDouble(), 0.0);
	auto cam1 = cameras.at(1);
}

FUNC(Test_JsonParser_overrides)
{
	std::istringstream stream1(R"(
{
	"Alpha": "RVS",
	"Beta" : [1.0, 2.0, 3.0],
	"Gamma" : 30
})");

	std::istringstream stream2(R"(
{
	"Gamma" : 40
})");

	auto root1 = json::Node::readFrom(stream1);
	auto root2 = json::Node::readFrom(stream2);

	CHECK(root1.require("Alpha").asString() == "RVS");
	EQUAL(root1.require("Gamma").asInt(), 30);

	root1.setOverrides(root2);

	CHECK(root1.require("Alpha").asString() == "RVS");
	EQUAL(root1.require("Gamma").asInt(), 40);
}

FUNC(Test_PoseTrace_loadFrom)
{
	std::istringstream stream(R"(X,Y,Z,Yaw,Pitch,Roll
0, 0, 0, 7.744e-06, 1.10991e-05, 2.98821e-06
-0.000289001, -4.49e-05, -2.50004e-05, -1337, 0.0172351, 0.00275806
-0.00062, -8.20999e-05, 1.0, 13.1, 42, 0.00589616
-0.047839, 0.0798379, -0.030835, -68.9187, -3.20608, 16.5

)");

	auto poseTrace = rvs::PoseTrace::loadFrom(stream);
	EQUAL(poseTrace.size(), 4u);
	EQUAL(poseTrace[0].position[0], 0.f);
	CHECK((poseTrace[1].position[1] + 4.49e-05) < 1e-12);
	EQUAL(poseTrace[2].position[2], 1.f);
	EQUAL(poseTrace[1].rotation[0], -1337.f);
	EQUAL(poseTrace[2].rotation[1], 42.f);
	EQUAL(poseTrace[3].rotation[2], 16.5f);
}

int main(int argc, const char* argv[])
{
	rvs::g_verbose = true;
	cv::setBreakOnError(true);
	return yaffut::main(argc, argv);
}
