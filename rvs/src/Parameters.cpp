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

#include "Parameters.hpp"
#include "PoseTraces.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace rvs
{
	namespace ProjectionType {
		int get_number(std::string proj) {
			if (proj == ProjectionType::perspective)
				return (int)ProjectionNumber::perspective;
			if (proj == ProjectionType::equirectangular)
				return (int)ProjectionNumber::equirectangular;
			return -1;
		}
	}
	namespace
	{
		cv::Matx33f rotationMatrixFromRotationAroundX(float rx)
		{
			return cv::Matx33f(
				1.f, 0.f, 0.f,
				0.f, cos(rx), -sin(rx),
				0.f, sin(rx), cos(rx));
		}

		cv::Matx33f rotationMatrixFromRotationAroundY(float ry)
		{
			return cv::Matx33f(
				cos(ry), 0.f, sin(ry),
				0.f, 1.f, 0.f,
				-sin(ry), 0.f, cos(ry));
		}

		cv::Matx33f rotationMatrixFromRotationAroundZ(float rz)
		{
			return cv::Matx33f(
				cos(rz), -sin(rz), 0.f,
				sin(rz), cos(rz), 0.f,
				0.f, 0.f, 1.f);
		}

		cv::Matx33f EulerAnglesToRotationMatrix(cv::Vec3f rotation)
		{
			return
				rotationMatrixFromRotationAroundZ(rotation[0]) *
				rotationMatrixFromRotationAroundY(rotation[1]) *
				rotationMatrixFromRotationAroundX(rotation[2]);
		}
	}

	Parameters::Parameters(json::Node root)
		: m_root(root)
	{}

	Parameters Parameters::readFrom(json::Node root)
	{
		Parameters parameters(root);

		parameters.setProjectionFrom(root);
		parameters.setPositionFrom(root);
		parameters.setRotationFrom(root);
		parameters.setDepthRangeFrom(root);
		parameters.setHasInvalidDepth(root);
		parameters.setResolutionFrom(root);
		parameters.setBitDepthColorFrom(root);
		parameters.setBitDepthDepthFrom(root);
		parameters.setColorFormatFrom(root);
		parameters.setDepthColorFormatFrom(root);
		parameters.setHorRangeFrom(root);
		parameters.setVerRangeFrom(root);
		parameters.setCropRegionFrom(root);
		parameters.setFocalFrom(root);
		parameters.setPrinciplePointFrom(root);
		parameters.setDisplacementMethodFrom(root);
		parameters.setMultiDepthRangeFrom(root);
		Parameters::validateUnused(root);


		return parameters;
	}

	json::Node const& Parameters::getRoot() const
	{
		return m_root;
	}

	std::string const& Parameters::getProjectionType() const
	{
		return m_projectionType;
	}

	cv::Vec3f Parameters::getRotation() const
	{
		return m_rotation;
	}

	void Parameters::setRotation(cv::Vec3f rotation)
	{
		m_rotation = rotation;
	}

	cv::Matx33f Parameters::getRotationMatrix() const
	{
		auto const radperdeg = 0.01745329252f;
		return EulerAnglesToRotationMatrix(radperdeg * m_rotation);
	}

	cv::Vec3f Parameters::getPosition() const
	{
		return m_position;
	}

	void Parameters::setPosition(cv::Vec3f position)
	{
		m_position = position;
	}

	cv::Vec2f Parameters::getDepthRange() const
	{
		return m_depthRange;
	}

	cv::Vec2f Parameters::getMultiDepthRange() const
	{
		return m_multidepthRange;
	}

	bool Parameters::hasInvalidDepth() const
	{
		return m_hasInvalidDepth;
	}

	cv::Size Parameters::getPaddedSize() const
	{
		return m_resolution;
	}

	cv::Size Parameters::getSize() const
	{
		return m_cropRegion.size();
	}

	cv::Rect Parameters::getCropRegion() const
	{
		return m_cropRegion;
	}

	int Parameters::getColorBitDepth() const
	{
		return m_bitDepthColor;
	}

	int Parameters::getDepthBitDepth() const
	{
		return m_bitDepthDepth;
	}

	ColorFormat Parameters::getColorFormat() const
	{
		return m_colorFormat;
	}

	ColorFormat Parameters::getDepthColorFormat() const
	{
		return m_depthColorFormat;
	}

	cv::Vec2f Parameters::getHorRange() const
	{
		assert(m_projectionType == ProjectionType::equirectangular);
		return m_horRange;
	}

	cv::Vec2f Parameters::getVerRange() const
	{
		assert(m_projectionType == ProjectionType::equirectangular);
		return m_verRange;
	}

	bool Parameters::isFullHorRange() const
	{
		return m_isFullHorRange;
	}

	cv::Vec2f Parameters::getFocal() const
	{
		assert(m_projectionType == ProjectionType::perspective);
		return m_focal;
	}

	void Parameters::setFocal(cv::Vec2f f)
	{
		m_focal = f;
	}

	cv::Vec2f Parameters::getPrinciplePoint() const
	{
		assert(m_projectionType == ProjectionType::perspective);
		return m_principlePoint - cv::Vec2f(cv::Point2f(m_cropRegion.tl()));
	}

	void Parameters::setPrinciplePoint(cv::Vec2f p)
	{
		m_principlePoint = p;
	}

    DisplacementMethod Parameters::getDisplacementMethod() const
    {
        return m_displacementMethod;
    }

	void Parameters::printTo(std::ostream& stream) const
	{
		stream << m_resolution << ' ' << m_bitDepthColor << "b " << m_bitDepthDepth << "b " << m_depthRange;
		if (m_cropRegion != cv::Rect(cv::Point(), m_resolution)) {
			stream << m_cropRegion;
		}
		stream << ' ' << m_projectionType << ' ';
		if (m_projectionType == ProjectionType::equirectangular) {
			stream << m_horRange << ' ' << m_verRange;
		}
		if (m_projectionType == ProjectionType::perspective) {
			stream << m_focal << ' ' << m_principlePoint;
		}
		stream << ' ' << m_position << ' ' << m_rotation;
	}

	void Parameters::setProjectionFrom(json::Node root)
	{
		m_projectionType = root.require("Projection").asString();
	}

	void Parameters::validateUnused(json::Node root)
	{
		if (root.require("Depthmap").asInt() != 1) {
			throw std::runtime_error("This version of RVS only supports Depthmap 1");
		}

		// NOTE: This field is not used consistently by the 3DoF+ test material.
		//       For RVS instead use the order of the camera names to determine the blending order.
		root.require("Background").asInt();
	}

	namespace
	{
		template<int N> cv::Vec<float, N> asFloatVec(json::Node node)
		{
			if (node.size() != N) {
				throw std::runtime_error("JSON parser: Expected a vector of floats");
			}
			cv::Vec<float, N> v;
			for (int i = 0; i != N; ++i) {
				v[i] = node.at(i).asFloat();
			}
			return v;
		}

		template<int N> cv::Vec<int, N> asIntVec(json::Node node)
		{
			if (node.size() != N) {
				throw std::runtime_error("JSON parser: Expected a vector of floats");
			}
			cv::Vec<int, N> v;
			for (int i = 0; i != N; ++i) {
				v[i] = node.at(i).asInt();
			}
			return v;
		}
	}

	void Parameters::setPositionFrom(json::Node root)
	{
		m_position = asFloatVec<3>(root.require("Position"));
	}

	void Parameters::setRotationFrom(json::Node root)
	{
		m_rotation = asFloatVec<3>(root.require("Rotation"));
	}

	void Parameters::setDepthRangeFrom(json::Node root)
	{
		m_depthRange = asFloatVec<2>(root.require("Depth_range"));
		if (!(m_depthRange[0] < m_depthRange[1])) {
			throw std::runtime_error("Inverted depth range: [near, far] with near > far is not allowed");
		}
		if (m_depthRange[1] > 1000.f) {
			throw std::runtime_error("Invalid depth range: [near, far] with far > 1000 is not allowed because 1000 stands for infinity. Please restrict depth range or change world units.");
		}
	}
	void Parameters::setMultiDepthRangeFrom(json::Node root)
	{
		auto node = root.optional("Multi_depth_range");

		if (node)
			m_multidepthRange = asFloatVec<2>(node);
		else
			m_multidepthRange = m_depthRange;
		if (!(m_multidepthRange[0] < m_multidepthRange[1])) {
			throw std::runtime_error("Inverted depth range: [near, far] with near > far is not allowed");
		}
	}

	void Parameters::setHasInvalidDepth(json::Node root)
	{
		// Backwards compatible with RVS 2
		m_hasInvalidDepth = true;

		auto node = root.optional("HasInvalidDepth");
		if (node) {
			m_hasInvalidDepth = node.asBool();
		}
	}

	void Parameters::setResolutionFrom(json::Node root)
	{
		m_resolution = cv::Size(asIntVec<2>(root.require("Resolution")));
	}

	void Parameters::setBitDepthColorFrom(json::Node root)
	{
		m_bitDepthColor = root.require("BitDepthColor").asInt();
	}

	void Parameters::setBitDepthDepthFrom(json::Node root)
	{
		m_bitDepthDepth = root.require("BitDepthDepth").asInt();
	}

	void Parameters::setColorFormatFrom(json::Node root)
	{
		auto colorFormat = root.require("ColorSpace").asString();

		if (colorFormat == "YUV420") {
			m_colorFormat = ColorFormat::YUV420;
		}
		else
			throw std::runtime_error("This version of RVS only supports YUV420 color space for texture");
	}

	void Parameters::setDepthColorFormatFrom(json::Node root)
	{
		auto depthColorFormat = root.require("DepthColorSpace").asString();

		if (depthColorFormat == "YUV420") {
			m_depthColorFormat = ColorFormat::YUV420;
		}
		else if (depthColorFormat == "YUV400") {
			m_depthColorFormat = ColorFormat::YUV400;
		}
		else
			throw std::runtime_error("This version of RVS only supports YUV420 and YUV400 color space for depth");
	}

	void Parameters::setHorRangeFrom(json::Node root)
	{
		if (m_projectionType == ProjectionType::equirectangular) {
			m_horRange = asFloatVec<2>(root.require("Hor_range"));

			m_isFullHorRange = false;
			try {
				m_isFullHorRange = asIntVec<2>(root.require("Hor_range")) == cv::Vec2i(-180, 180);
			}
			catch (std::runtime_error&) {}
		}
	}

	void Parameters::setVerRangeFrom(json::Node root)
	{
		if (m_projectionType == ProjectionType::equirectangular) {
			m_verRange = asFloatVec<2>(root.require("Ver_range"));
		}
	}

	void Parameters::setCropRegionFrom(json::Node root)
	{
		auto node = root.optional("Crop_region");
		if (node) {
			auto values = asIntVec<4>(node);
			m_cropRegion = cv::Rect(values[0], values[1], values[2], values[3]);
		}
		else {
			m_cropRegion = cv::Rect(cv::Point(), m_resolution);
		}
	}

	void Parameters::setFocalFrom(json::Node root)
	{
		if (m_projectionType == ProjectionType::perspective) {
			m_focal = asFloatVec<2>(root.require("Focal"));
		}
	}

	void Parameters::setPrinciplePointFrom(json::Node root)
	{
		if (m_projectionType == ProjectionType::perspective) {
			m_principlePoint = asFloatVec<2>(root.require("Principle_point"));
		}
	}


	void Parameters::setDisplacementMethodFrom(json::Node root){
	    auto node = root.optional("DisplacementMethod");
		if (node) {
        		if (node.asString()=="Depth")
                m_displacementMethod = DisplacementMethod::depth;
			if (node.asString()=="Polynomial")
                m_displacementMethod = DisplacementMethod::polynomial;
		}
		else {
        //        std::cout << "DisplacementMethod: " << "Depth (default)" << std::endl;
                m_displacementMethod = DisplacementMethod::depth;
		}
	}

}
