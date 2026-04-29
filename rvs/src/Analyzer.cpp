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

#include "Analyzer.hpp"
#include "SynthesizedView.hpp"
#include "BlendedView.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace rvs
{
	namespace
	{
		void dump(std::string basename, int inputFrame, int inputView, int virtualFrame, int virtualView, View const& view)
		{
			cv::Mat3f rgb;
			cv::cvtColor(view.get_color(), rgb, cv::COLOR_YCrCb2BGR);
			cv::Mat3b color;
			rgb.convertTo(color, CV_8U, 255.);
			std::ostringstream filepath;
			filepath.str(""); filepath << basename << "-color-" << inputFrame << '.' << inputView << "to" << virtualFrame << '.' << virtualView << ".png";
			cv::imwrite(filepath.str(), color);

			filepath.str(""); filepath << basename << "-quality-" << inputFrame << '.' << inputView << "to" << virtualFrame << '.' << virtualView << ".exr";
			cv::imwrite(filepath.str(), view.get_quality());

			filepath.str(""); filepath << basename << "-depth-" << inputFrame << '.' << inputView << "to" << virtualFrame << '.' << virtualView << ".exr";
			cv::imwrite(filepath.str(), view.get_depth());

			filepath.str(""); filepath << basename << "-validity-" << inputFrame << '.' << inputView << "to" << virtualFrame << '.' << virtualView << ".exr";
			cv::imwrite(filepath.str(), view.get_validity());

			filepath.str(""); filepath << basename << "-depth_mask-" << inputFrame << '.' << inputView << "to" << virtualFrame << '.' << virtualView << ".png";
			cv::imwrite(filepath.str(), view.get_depth_mask());
		}
	}

	Analyzer::Analyzer(std::string const& filepath)
		: Application(filepath)
	{}

	void Analyzer::onIntermediateSynthesisResult(int inputFrame, int inputView, int virtualFrame, int virtualView, SynthesizedView const& synthesizedView)
	{
		dump("warping", inputFrame, inputView, virtualFrame, virtualView, synthesizedView);
	}

	void Analyzer::onIntermediateBlendingResult(int inputFrame, int inputView, int virtualFrame, int virtualView, BlendedView const& blendedView)
	{
		dump("blending", inputFrame, inputView, virtualFrame, virtualView, blendedView);
	}
}