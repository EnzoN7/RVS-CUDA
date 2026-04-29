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

#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "Pipeline.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
@file Application.hpp
\brief The file containing the regular RVS application
*/

namespace rvs
{
	/**
	\brief The main class of the regular RVS application

	The application executes the following steps:
		- Parsing the configuration file (see Parser)
		- Loading the input reference views (see InputView, and load_images());
		- View synthesis of the target view once for each input reference view (see SynthesizedView);
		- Blending all the SynthesizedView together by assigning a per-pixel quality to each synthesized view (see BlendedView);
		- Inpainting to fill the remaining holes (see inpaint());
		- Writing the output (see write_color()).
	*/
	class Application : public Pipeline
	{
	public:
		/**
		\brief Constructor

		Please note that Application itself also calls getConfig() to allow derived classees overriding the configuration.

		@param filepath Configuration file (JSON format)
		*/
		Application(std::string const& filepath, std::string const& sourcepath = "");

	protected:
		Config const& getConfig() const override;

		std::shared_ptr<View> Application::loadInputView(int inputFrame, int inputView, Parameters const& parameters) override;

		bool wantColor() override;
		bool wantMaskedColor() override;
		bool wantMask() override;
		bool wantDepth() override;
		bool wantMaskedDepth() override;

		void saveColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters) override;
		void saveColor(float3* color, int virtualFrame, int virtualView, Parameters const& parameters, cv::Size size) override;
		void saveMaskedColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters) override;
		void saveMask(cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const& parameters) override;
		void saveDepth(cv::Mat1f depth, int virtualFrame, int virtualView, Parameters const& parameters) override;
		void saveMaskedDepth(cv::Mat1f depth, cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const& parameters) override;

	private:
		Config m_config;
	};
}

#endif
