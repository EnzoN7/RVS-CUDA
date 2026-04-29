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

#include "View.hpp"
#include "inpainting.hpp"
#include "image_loading.hpp"

#include <fstream>
#include <iostream>
#include <future>

namespace rvs
{
	// Initialize all maps at once
	View::View(cv::Mat3f color, cv::Mat1f depth, cv::Mat1f quality, cv::Mat1f validity)
	{
		assign(color, depth, quality, validity);
	}

	View::~View() {}

	// Initialize all maps at once
	void View::assign(cv::Mat3f color, cv::Mat1f depth, cv::Mat1f quality, cv::Mat1f validity)
	{
		m_color = color;
		m_depth = depth;
		m_quality = quality;
		m_validity = validity;
		validate();
	}
	void View::assign(float3* dev_color, float* dev_depth, cv::Mat1f quality, cv::Mat1f validity)
	{
		m_dev_color = dev_color;
		m_dev_depth = dev_depth;
		m_quality = quality;
		m_validity = validity;
	}
	void View::assign(cv::Mat3f color, cv::Mat1f depth, cv::Mat1f quality, cv::Mat1f validity, PolynomialDepth polynomial_depth)
	{
		m_color = color;
		m_depth = depth;
		m_quality = quality;
		m_validity = validity;
		m_polynomial_depth = polynomial_depth;
		validate();
	}
	void View::assign(float3* dev_color, cv::Mat1f depth, cv::Mat1f quality, cv::Mat1f validity, PolynomialDepth polynomial_depth)
	{
		m_dev_color = dev_color;
		m_depth = depth;
		m_quality = quality;
		m_validity = validity;
		m_polynomial_depth = polynomial_depth;
		validate();
	}

	/** For GPU optimization */
	void View::assign(float3* dev_color, float* dev_depth, float* dev_validity, cv::Size output_size)
	{
		m_dev_color = dev_color;
		m_dev_depth = dev_depth;
		m_dev_validity = dev_validity;
		m_output_size = output_size;
	}

	void View::assign(float3* dev_color, float* dev_depth, float* dev_validity)
	{
		m_dev_color = dev_color;
		m_dev_depth = dev_depth;
		m_dev_validity = dev_validity;
	}

	void View::assign(cv::Size output_size)
	{
		m_output_size = output_size;
	}

	float3* View::getDevColor() const
	{
		return m_dev_color;
	}

	float* View::getDevDepth() const
	{
		return m_dev_depth;
	}

	float* View::getDevValidity() const
	{
		return m_dev_validity;
	}

	cv::Size View::getOutputSize() const
	{
		return m_output_size;
	}

	// Return the texture
	cv::Mat3f View::get_color() const
	{
		validate();
		CV_Assert(!m_color.empty());
		return m_color;
	}

	// Return the depth map (same size as texture)
	cv::Mat1f View::get_depth() const
	{
		validate();
		CV_Assert(!m_depth.empty());
		return m_depth;
	}

	PolynomialDepth View::get_polynomial_depth() const
	{
		validate();
		return m_polynomial_depth;
	}

	DisplacementMethod InputView::get_displacementMethod() const
	{
		return parameters.getDisplacementMethod();
	}

	// Return the quality map (same size as texture)
	cv::Mat1f View::get_quality() const
	{
		validate();
		CV_Assert(!m_quality.empty());
		return m_quality;
	}

	// Return the validity map (same size as texture)
	cv::Mat1f View::get_validity() const
	{
		validate();
		CV_Assert(!m_validity.empty());
		return m_validity;
	}

	// Return the size of the texture and depth map
	cv::Size View::get_size() const
	{
		validate();
		return m_color.size();
	}

	// Return a mask with all valid depth values
	cv::Mat1b View::get_depth_mask() const
	{
		return get_depth() > 0.f; // excludes NaN's
	}

	// Calculate a mask for inpainting
	cv::Mat1b View::get_inpaint_mask() const
	{
		auto inpaint_mask = cv::Mat1b(get_size(), 255);
		inpaint_mask.setTo(0, get_quality() > 0.f); // excludes NaN's
		return inpaint_mask;
	}

	// Calculate a mask with valid pixels for masked output
	cv::Mat1b View::get_validity_mask(float threshold) const
	{
		auto validity_mask = cv::Mat1b(get_size(), 255);
		validity_mask.setTo(0, get_validity() > threshold); // excludes NaN's
		return validity_mask;
	}

	void View::validate() const
	{
		//auto size = m_color.size();
		//CV_Assert(m_depth.empty() || m_depth.size() == size);
		//CV_Assert(m_quality.empty() || m_quality.size() == size);
		//CV_Assert(m_validity.size() == m_quality.size());
	}

	// Load a color image and depth map
	InputView::InputView(std::string const& filepath_color, std::string const& filepath_depth, int frame, Parameters const& parameters)
		:
		filepath_color(filepath_color),
		filepath_depth(filepath_depth),
		frame(frame),
		parameters(parameters)
	{
		load();
	}

	// Load a color image and depth map
	void InputView::load()
	{
		if (parameters.getDisplacementMethod() == DisplacementMethod::depth)
		{
			std::future<float3*> color_future = std::async(std::launch::async, read_color, filepath_color, frame, parameters);
			std::future<float*> depth_future = std::async(std::launch::async, read_depth, filepath_depth, frame, parameters);

			float3* col = color_future.get();
			float* dep = depth_future.get();

			assign(col, dep, cv::Mat1f(), cv::Mat1f());
		}
		else if (parameters.getDisplacementMethod()==DisplacementMethod::polynomial)
		{
            assign
			(
                read_color(filepath_color, frame, parameters),
                cv::Mat1f::zeros(parameters.getSize()),
                cv::Mat1f(),
                cv::Mat1f(),
                read_polynomial_depth(filepath_depth, frame, parameters)
			);
		}
		loaded = true;
	}

	void InputView::unload()
	{
		assign(cv::Mat3f(), cv::Mat1f(), cv::Mat1f(), cv::Mat1f());
		loaded = false;
	}
}
