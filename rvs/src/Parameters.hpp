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

#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_

#include "JsonParser.hpp"
#include <opencv2/core.hpp>

namespace rvs
{
	enum class ProjectionNumber
	{
		perspective,
		equirectangular
	};

	namespace ProjectionType
	{
		auto const perspective = "Perspective";
		auto const equirectangular = "Equirectangular";
		int get_number(std::string proj);
	}

	enum class ColorFormat
	{
		YUV420
	};

	enum class DisplacementMethod
	{
		depth
	};

	class Parameters
	{
	  public:
		static Parameters readFrom(json::Node parameters);
		json::Node const& getRoot() const;
		std::string const& getProjectionType() const;

		cv::Vec3f getRotation() const;
		void setRotation(cv::Vec3f);

		cv::Matx33f getRotationMatrix() const;

		cv::Vec3f getPosition() const;
		void setPosition(cv::Vec3f);

		cv::Vec2f getDepthRange() const;
		cv::Vec2f getMultiDepthRange() const;
		bool hasInvalidDepth() const;
		cv::Size getPaddedSize() const;
		cv::Size getSize() const;
		cv::Rect getCropRegion() const;
		int getColorBitDepth() const;
		int getDepthBitDepth() const;
		ColorFormat getColorFormat() const;
		ColorFormat getDepthColorFormat() const;
		cv::Vec2f getHorRange() const;
		cv::Vec2f getVerRange() const;
		bool isFullHorRange() const;

		cv::Vec2f getFocal() const;
		void setFocal(cv::Vec2f f);

		cv::Vec2f getPrinciplePoint() const;
		void setPrinciplePoint(cv::Vec2f p);

		void printTo(std::ostream& stream) const;

		DisplacementMethod getDisplacementMethod() const;

	  private:
		Parameters(json::Node root);

		void setProjectionFrom(json::Node root);
		void setPositionFrom(json::Node root);
		void setRotationFrom(json::Node root);
		void setDepthRangeFrom(json::Node root);
		void setMultiDepthRangeFrom(json::Node root);
		void setHasInvalidDepth(json::Node root);
		void setResolutionFrom(json::Node root);
		void setBitDepthColorFrom(json::Node root);
		void setBitDepthDepthFrom(json::Node root);
		void setColorFormatFrom(json::Node root);
		void setDepthColorFormatFrom(json::Node root);
		void setHorRangeFrom(json::Node root);
		void setVerRangeFrom(json::Node root);
		void setCropRegionFrom(json::Node root);
		void setFocalFrom(json::Node root);
		void setPrinciplePointFrom(json::Node root);
		void setDisplacementMethodFrom(json::Node root);

		static void validateUnused(json::Node root);

		json::Node m_root;
		std::string m_projectionType;
		cv::Vec3f m_position;
		cv::Vec3f m_rotation;
		cv::Vec2f m_depthRange;
		cv::Vec2f m_multidepthRange;
		bool m_hasInvalidDepth;
		cv::Size m_resolution;
		int m_bitDepthColor;
		int m_bitDepthDepth;
		ColorFormat m_colorFormat;
		ColorFormat m_depthColorFormat;
		cv::Vec2f m_horRange;
		cv::Vec2f m_verRange;
		bool m_isFullHorRange;
		cv::Rect m_cropRegion;
		cv::Vec2f m_focal;
		cv::Vec2f m_principlePoint;
		DisplacementMethod m_displacementMethod;
	};
}

#endif
