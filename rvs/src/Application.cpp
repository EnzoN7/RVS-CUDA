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

#include "Application.hpp"
#include "image_writing.hpp"
#include <iostream>

namespace rvs
{
	Application::Application(std::string const& filepath, std::string const& sourcepath )
		: m_config(Config::loadFromFile(filepath))
	{
        
        if( !sourcepath.empty() )
        {
            for( auto& name : m_config.texture_names )
                name = sourcepath + "/"+ name;
                
            for( auto& name : m_config.depth_names )
                name = sourcepath + "/" + name;
        }
    }

	Config const& Application::getConfig() const
	{
		return m_config;
	}

	std::shared_ptr<View> Application::loadInputView(int inputFrame, int inputView, Parameters const& parameters)
	{
		return std::make_shared<InputView>
		(
			getConfig().texture_names[inputView],
			getConfig().depth_names[inputView],
			inputFrame,
			parameters
		);
	}

	bool Application::wantColor()
	{
		return !getConfig().outfilenames.empty();
	}

	bool Application::wantMaskedColor()
	{
		return !getConfig().outmaskedfilenames.empty();
	}

	bool Application::wantMask()
	{
		return !getConfig().outmaskfilenames.empty();
	}

	bool Application::wantDepth()
	{
		return !getConfig().outdepthfilenames.empty();
	}

	bool Application::wantMaskedDepth()
	{
		return !getConfig().outmaskdepthfilenames.empty();
	}

	void Application::saveColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters)
	{
		write_color(getConfig().outfilenames[virtualView], color, virtualFrame, parameters);
	}
	
	void Application::saveColor(float3* devColor, int virtualFrame, int virtualView, Parameters const& parameters, cv::Size size)
	{
		write_color(getConfig().outfilenames[virtualView], devColor, virtualFrame, parameters, size);
	}

	void Application::saveMaskedColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters)
	{
		write_color(getConfig().outmaskedfilenames[virtualView], color, virtualFrame, parameters);
	}


	void Application::saveMask(cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const& parameters)
	{
		write_mask(getConfig().outmaskfilenames[virtualView], mask, virtualFrame, parameters);
	}

	void Application::saveDepth(cv::Mat1f depth, int virtualFrame, int virtualView, Parameters const & parameters)
	{
		write_depth(getConfig().outdepthfilenames[virtualView], depth, virtualFrame, parameters);
	}

	void Application::saveMaskedDepth(cv::Mat1f depth, cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const & parameters)
	{
		write_maskedDepth(getConfig().outmaskdepthfilenames[virtualView], depth, mask, virtualFrame, parameters);
	}
}
