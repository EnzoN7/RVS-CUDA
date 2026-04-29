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

#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_

#include "JsonParser.hpp"
#include <opencv2/core.hpp>

/**
@file Parameters.hpp
\brief Definition of extrinsic and intrinsic camera parameters as well as video parameters
*/

namespace rvs
{
	/**\brief Projection type

	\see Projector*/
	enum class ProjectionNumber {
		perspective,
		equirectangular
	};

	namespace ProjectionType {
		auto const perspective = "Perspective";
		auto const equirectangular = "Equirectangular";
		int get_number(std::string proj);
	}

	enum class ColorFormat {
		YUV420,
		YUV400
	};

	enum class DisplacementMethod {
		depth,
		polynomial
	};

	/** Camera and video parameters */
	class Parameters {
	public:
		/** Camera parameters
		@param parameters The camera and video parameters of this camera
		*/
		static Parameters readFrom(json::Node parameters);

		/** Direct access to the parameter set for the software platform and proposals */
		json::Node const& getRoot() const;

		/** The projection type */
		std::string const& getProjectionType() const;

		/** Extrinsic parameter of rotation (Euler angles, degrees) */
		cv::Vec3f getRotation() const;

		/** Set rotation (Euler angles, degrees) */
		void setRotation(cv::Vec3f);

		/** Get rotation as a matrix */
		cv::Matx33f getRotationMatrix() const;

		/** Extrinsic parameter of translation */
		cv::Vec3f getPosition() const;

		/** Set a new translation */
		void setPosition(cv::Vec3f);

		/** Depth range

		perspective: [znear, zfar]
		equirectangular: [Rmin, Rmax] */
		cv::Vec2f getDepthRange() const;

		/** MultiDepth range */
		cv::Vec2f getMultiDepthRange() const;

		/** Has invalid depth flag */
		bool hasInvalidDepth() const;

		/** Padded image size (before cropping) */
		cv::Size getPaddedSize() const;

		/** Image size after cropping */
		cv::Size getSize() const;

		/** Image parameter of crop region:

		perspective: principle point relates to uncropped region
		equirectangular: angular ranges relate to cropped region */
		cv::Rect getCropRegion() const;

		/** Texture bit depth */
		int getColorBitDepth() const;

		/** Depth map bit depth */
		int getDepthBitDepth() const;

		/** Color space */
		ColorFormat getColorFormat() const;

		/** Depth color space */
		ColorFormat getDepthColorFormat() const;

		/** Horizontal angular range (degrees) */
		cv::Vec2f getHorRange() const;

		/** Vertical angular range (degrees) */
		cv::Vec2f getVerRange() const;

		/** Is full horizontal angular range? */
		bool isFullHorRange() const;

		/** Intrinsic parameter of focal length (perspective) */
		cv::Vec2f getFocal() const;
		void setFocal(cv::Vec2f f);

		/** Intrinsic parameter of principle point (perspective)

		The value returned is already adjusted to refer to the cropped region. */
		cv::Vec2f getPrinciplePoint() const;
		void setPrinciplePoint(cv::Vec2f p);

		/** Print a description */
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

		/** Validate some fields of the JSON format that RVS is not effectively using */
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
