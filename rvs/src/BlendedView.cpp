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
	Sarah Fachada Sarah.Fernandes.Pinto.Fachada@ulb.ac.be
	Daniele Bonatto Daniele.Bonatto@ulb.ac.be
	Arnaud Schenkel arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
	Bart Kroon, bart.kroon@philips.com
	Bart Sonneveldt, bart.sonneveldt@philips.com
*/

#include "BlendedView.hpp"
#include "blending.hpp"
#include "blend_img.cuh"

#include <iostream>
#include <chrono>

#if WITH_OPENGL
#include "helpersGL.hpp"
#include "RFBO.hpp"
#endif

namespace rvs
{
	BlendedView::~BlendedView() {}

	BlendedViewMultiSpec::BlendedViewMultiSpec(float exp_low_freq, float exp_high_freq)
		: m_low_freq(exp_low_freq)
		, m_high_freq(exp_high_freq)
	{}

	BlendedViewMultiSpec::~BlendedViewMultiSpec() {}

	void BlendedViewMultiSpec::blend(View const& view)
	{
		// Split color frequencies in low and high band
		cv::Mat3f low_color;
		cv::Mat3f high_color;
		auto mask = cv::Mat1b(view.get_quality() > 0.f);
		detail::split_frequencies(view.get_color(), low_color, high_color, mask);

		// Repack as views
		auto low_view = View(low_color, view.get_depth(), view.get_quality(), view.get_validity());
		auto high_view = View(high_color, view.get_depth(), view.get_quality(), view.get_validity());

		// Blend low and high frequency separately
		m_low_freq.blend(low_view);
		m_high_freq.blend(high_view);

		// Combine result
		assign(m_low_freq.get_color() + m_high_freq.get_color(), m_low_freq.get_depth(), m_low_freq.get_quality(), m_low_freq.get_validity());
	}

	void BlendedViewMultiSpec::freeMemory(cv::Size imgSize)
	{
		int i = imgSize.area();
		i++;
		return;
	}

	BlendedViewSimple::BlendedViewSimple(float blending_exp)
		: m_is_empty(true)
		, m_blending_exp(blending_exp)
	{
		assign(cv::Mat3f(), cv::Mat1f(), cv::Mat1f(), cv::Mat1f());
	}

	BlendedViewSimple::~BlendedViewSimple() {}

	const cv::Vec3f empty_rgb_color(0.0f, 1.0f, 0.0f);

	void BlendedViewSimple::blendBasicMode(View const& view)
	{
#if WITH_OPENGL
		if (g_with_opengl) {
			auto FBO = opengl::RFBO::getInstance();
			auto& shaders = opengl::ShadersList::getInstance();
			// QUAD VAO
			GLfloat quadVertices[] = {
				-1.0f, 1.0f, 0.0f, 1.0f,
				-1.0f, -1.0f, 0.0f, 0.0f,
				1.0f, -1.0f, 1.0f, 0.0f,
				-1.0f, 1.0f, 0.0f, 1.0f,
				1.0f, -1.0f, 1.0f, 0.0f,
				1.0f, 1.0f, 1.0f, 1.0f };
			GLuint quadVAO, quadVBO;
			glGenVertexArrays(1, &quadVAO);
			glGenBuffers(1, &quadVBO);
			glBindVertexArray(quadVAO);
			glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
				4 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
				4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(GLfloat)));
			glBindVertexArray(0);
			// END QUAD VAO

			GLuint program = shaders("blending2").getProgramID();
			if (FBO->value) program = shaders("blending1").getProgramID();

			assert(program != 0);

			glBindFramebuffer(GL_FRAMEBUFFER, FBO->ID);
			//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glUseProgram(program);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, FBO->image);
			glUniform1i(glGetUniformLocation(program, "new_image"), 0);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, FBO->depth);
			glUniform1i(glGetUniformLocation(program, "new_depth"), 1);
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, FBO->quality_triangle);
			glUniform1i(glGetUniformLocation(program, "new_triangle_quality"), 2);

			glActiveTexture(GL_TEXTURE3 + 2 * FBO->value);
			glBindTexture(GL_TEXTURE_2D, FBO->swap_image[FBO->value]);
			glUniform1i(glGetUniformLocation(program, "accumulator_image"), 3 + 2 * FBO->value);

			glActiveTexture(GL_TEXTURE4 + 2 * FBO->value);
			glBindTexture(GL_TEXTURE_2D, FBO->swap_quality[FBO->value]);
			glUniform1i(glGetUniformLocation(program, "accumulator_quality"), 4 + 2 * FBO->value);

			// parameters
			glUniform1f(glGetUniformLocation(program, "blending_factor"), m_blending_exp);
			glUniform1f(glGetUniformLocation(program, "input_distance"), static_cast<float>(view.distance_from_origin));
			// end parameters

			glBindVertexArray(quadVAO);
			glDrawArrays(GL_TRIANGLES, 0, 6);
			glUseProgram(0);
			glBindVertexArray(0);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			FBO->toggle();
		}
#endif
		if (!g_with_opengl) {
			if (m_is_empty)
			{
				m_is_empty = false;
				assign(view.get_color(), view.get_depth(), view.get_quality(), view.get_validity());
				m_depth_mask = view.get_depth_mask();
			}
			else
			{
				std::vector<cv::Mat> colors = { get_color(), view.get_color() };
				std::vector<cv::Mat> qualities = { get_quality(), view.get_quality() };
				std::vector<cv::Mat> depth_masks = { m_depth_mask, view.get_depth_mask() };

				cv::Mat quality;
				cv::Mat depth_prolongation_mask;
				cv::Mat inpaint_mask; // don't care
				cv::Mat color;

				color = detail::blend_img(
					colors, qualities, depth_masks,
					empty_rgb_color,
					/*out*/ quality,
					/*out*/ depth_prolongation_mask,
					/*out*/ inpaint_mask,
					m_blending_exp);

				auto depth = get_depth().clone();
				view.get_depth().copyTo(depth, view.get_quality() > get_quality());

				assign(color, depth, quality, max(get_validity(), view.get_validity()));
				m_depth_mask = depth_prolongation_mask;
			}
		}
	}

	void BlendedViewSimple::blendCudaMode(View const& view)
	{
		if (m_is_empty)
		{
			m_is_empty = false;

			assign(view.getDevColor(), view.getDevDepth(), view.getDevValidity());
		}
		else
		{
			try
			{
				float* devOutputValidity;
				float* devOutputDepth;
				float3* devOutputColor;

				blendImages
				(
					getDevColor(),
					getDevValidity(),
					getDevDepth(),
					view.getDevColor(),
					view.getDevValidity(),
					view.getDevDepth(),
					view.getOutputSize(),
					empty_rgb_color,
					devOutputColor,
					devOutputValidity,
					devOutputDepth,
					m_blending_exp
				);

				assign(devOutputColor, devOutputDepth, devOutputValidity);
			}
			catch (const std::runtime_error& e)
			{
				std::cerr << "CUDA Error (blend) : " << e.what() << std::endl;
			};
		}
	}

	void BlendedViewSimple::blend(View const& view)
	{
		if (g_with_cuda)
			blendCudaMode(view);
		else
			blendBasicMode(view);
	}

	void BlendedViewSimple::freeMemory(cv::Size imgSize)
	{
		cv::Mat1f outputValidity;
		cv::Mat1f outputDepth;

		cudaFree(getDevValidity());
		cudaFree(getDevDepth());

		assign(imgSize);
	}

#if WITH_OPENGL
	void BlendedView::assignFromGL2CV(cv::Size size)
	{
		auto FBO = opengl::RFBO::getInstance();
		cv::Mat3f img(size, CV_32FC3);

		glPixelStorei(GL_PACK_ALIGNMENT, 4);
		glPixelStorei(GL_PACK_ROW_LENGTH, GLint(img.step / img.elemSize()));

		glBindFramebuffer(GL_FRAMEBUFFER, FBO->ID);
		glReadBuffer(GL_COLOR_ATTACHMENT3 + 2 * FBO->value);
		glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_FLOAT, img.ptr());

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		cv::flip(img, img, 0);

		cv::Mat validity = cv::Mat::ones(size, CV_32FC1);

		glPixelStorei(GL_PACK_ALIGNMENT, 4);
		glPixelStorei(GL_PACK_ROW_LENGTH, GLint(validity.step / validity.elemSize()));

		glBindFramebuffer(GL_FRAMEBUFFER, FBO->ID);
		glReadBuffer(GL_COLOR_ATTACHMENT4 + 2 * (FBO->value));
		glReadPixels(0, 0, validity.cols, validity.rows, GL_RED, GL_FLOAT, validity.ptr());

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		cv::flip(validity, validity, 0);

		assign(img, cv::Mat1f(), validity, validity);
		FBO->clear_buffers();
	}
#endif
}
