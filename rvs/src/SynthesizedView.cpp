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

#include "SynthesizedView.hpp"
#include "transform.hpp"
#include "transform_triangle.cuh"
#include "unproject_project.cuh"
#include "scale_uv.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>

#if WITH_OPENGL
#include "helpersGL.hpp"
#include "RFBO.hpp"
#endif

namespace rvs
{
	namespace
	{
		// Affine transformation: x -> Rx + t
		cv::Mat3f affine_transform(cv::Mat3f x, cv::Matx33f R, cv::Vec3f t)
		{
			auto y = cv::Mat3f(x.size());

			for (int i = 0; i != y.rows; ++i) {
				for (int j = 0; j != y.cols; ++j) {
					y(i, j) = R * x(i, j) + t;
				}
			}

			return y;
		}
	}

	SynthesizedView::SynthesizedView() {}

	SynthesizedView::~SynthesizedView() {}

	void SynthesizedView::computeBasicMode(View& input)
	{
		auto const& pu_transformer = static_cast<const GenericTransformer*>(m_space_transformer);
		cv::Size inputSize = pu_transformer->getInputParameters().getSize();
		cv::Mat3f inputColor(inputSize);
		cv::Mat1f inputDepth(inputSize);

		cudaMemcpy(inputColor.ptr<float3>(), input.getDevColor(), inputSize.area() * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(inputDepth.ptr<float>(), input.getDevDepth(), inputSize.area() * sizeof(float), cudaMemcpyDeviceToHost);

		assert(m_space_transformer);

		auto R = m_space_transformer->get_rotation();
		auto t = m_space_transformer->get_translation();
		distance_from_origin = cv::max(0.01, cv::norm(t));

#if WITH_OPENGL
		GLuint nl_mask_idx = 0;
#endif

		if (input.get_displacementMethod() == DisplacementMethod::polynomial) {
			cv::Mat1f newmask = cv::Mat1f::zeros(input.get_polynomial_depth().m_polynomial[19].size());
			cv::Vec3f disp = R * t;
			cv::Mat1f dispx = disp[1] * input.get_polynomial_depth().m_polynomial[9];
			cv::Mat1f dispy = disp[2] * input.get_polynomial_depth().m_polynomial[9];
			for (int y = 0; y < dispx.size().height; ++y) {
				for (int x = 0; x < dispx.size().height; ++x) {
					int dy = static_cast<int>(floor(y - dispy.at<float>(y, x)));
					int dx = static_cast<int>(floor(x - dispx.at<float>(y, x)));
					dy = cv::min(cv::max(0, dy), dispx.size().height - 1);
					dx = cv::min(cv::max(0, dx), dispx.size().width - 1);
					if (input.get_polynomial_depth().m_polynomial[19].at<float>(y, x) > 0.1)
						newmask.at<float>(dy, dx) = 1.0;
				}
			}
			cv::morphologyEx(newmask, newmask, cv::MORPH_CLOSE, cv::getStructuringElement(0, cv::Size(5, 5)), cv::Point(-1, -1), 2);
			cv::morphologyEx(newmask, newmask, cv::MORPH_OPEN, cv::getStructuringElement(0, cv::Size(5, 5)));
#if WITH_OPENGL
			if (g_with_opengl) {
				nl_mask_idx = opengl::cvMat2glTexture(newmask);
			}
#endif
		}

#if WITH_OPENGL
		if (g_with_opengl) {
			auto ogl_transformer = static_cast<const OpenGLTransformer*>(m_space_transformer);
			GLuint image_texture = opengl::cvMat2glTexture(inputColor);
			GLuint depth_texture = 0;
			GLuint mask_texture = 0;
			GLuint polynomial1_texture = 0;
			GLuint polynomial2_texture = 0;
			GLuint polynomial3_texture = 0;
			GLuint polynomial4_texture = 0;
			GLuint polynomial5_texture = 0;

			if (input.get_displacementMethod() == DisplacementMethod::depth)
			{
				depth_texture = opengl::cvMat2glTexture(inputDepth / input.get_max_depth());
			}
			if (input.get_displacementMethod() == DisplacementMethod::polynomial) {
				std::array<cv::Mat1f, 20> polynomial = input.get_polynomial_depth().m_polynomial;
				std::vector<cv::Mat1f> pol1 = { polynomial[0],polynomial[1],polynomial[2],polynomial[3] };
				std::vector<cv::Mat1f> pol2 = { polynomial[4],polynomial[5],polynomial[6],polynomial[7] };
				std::vector<cv::Mat1f> pol3 = { polynomial[8],polynomial[9],polynomial[10],polynomial[11] };
				std::vector<cv::Mat1f> pol4 = { polynomial[12],polynomial[13],polynomial[14],polynomial[15] };
				std::vector<cv::Mat1f> pol5 = { polynomial[16],polynomial[17],polynomial[18],polynomial[19] };
				cv::Mat4f p1, p2, p3, p4, p5;
				cv::merge(pol1, p1);
				cv::merge(pol2, p2);
				cv::merge(pol3, p3);
				cv::merge(pol4, p4);
				cv::merge(pol5, p5);
				depth_texture = opengl::cvMat2glTexture(polynomial[9]);
				mask_texture = opengl::cvMat2glTexture(polynomial[19]);
				polynomial1_texture = opengl::cvMat2glTexture(p1);
				polynomial2_texture = opengl::cvMat2glTexture(p2);
				polynomial3_texture = opengl::cvMat2glTexture(p3);
				polynomial4_texture = opengl::cvMat2glTexture(p4);
				polynomial5_texture = opengl::cvMat2glTexture(p5);
			}

			auto FBO = opengl::RFBO::getInstance();
			auto& shaders = opengl::ShadersList::getInstance();

			float w = float(inputSize.width);//.get_depth().cols);
			float h = float(inputSize.height);//.get_depth().rows);
			float n_w = float(ogl_transformer->getVirtualParameters().getSize().width);
			float n_h = float(ogl_transformer->getVirtualParameters().getSize().height);

			glm::vec3 translation = glm::vec3(t[0], t[1], t[2]);
			glm::mat3x3 rotation(0);
			opengl::fromCV2GLM<3, 3>(cv::Mat(R), &rotation);

			const opengl::VAO_VBO_EBO vve(inputDepth.size());

			GLuint program;
			if (input.get_displacementMethod() == DisplacementMethod::polynomial)
				program = shaders("synthesis_polynomial").getProgramID();
			else
				program = shaders("synthesis").getProgramID();
			assert(program != 0);

			glEnable(GL_DEPTH_TEST);

			glBindFramebuffer(GL_FRAMEBUFFER, FBO->ID);
			glClear(GL_DEPTH_BUFFER_BIT);

			glUseProgram(program);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, image_texture);
			glUniform1i(glGetUniformLocation(program, "image_texture"), 0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, depth_texture);
			glUniform1i(glGetUniformLocation(program, "depth_texture"), 1);

			if (input.get_displacementMethod() == DisplacementMethod::polynomial) {
				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, mask_texture);
				glUniform1i(glGetUniformLocation(program, "mask_texture"), 2);

				glActiveTexture(GL_TEXTURE3);
				glBindTexture(GL_TEXTURE_2D, polynomial1_texture);
				glUniform1i(glGetUniformLocation(program, "polynomial1_texture"), 3);

				glActiveTexture(GL_TEXTURE4);
				glBindTexture(GL_TEXTURE_2D, polynomial2_texture);
				glUniform1i(glGetUniformLocation(program, "polynomial2_texture"), 4);

				glActiveTexture(GL_TEXTURE5);
				glBindTexture(GL_TEXTURE_2D, polynomial3_texture);
				glUniform1i(glGetUniformLocation(program, "polynomial3_texture"), 5);

				glActiveTexture(GL_TEXTURE6);
				glBindTexture(GL_TEXTURE_2D, polynomial4_texture);
				glUniform1i(glGetUniformLocation(program, "polynomial4_texture"), 6);

				glActiveTexture(GL_TEXTURE7);
				glBindTexture(GL_TEXTURE_2D, polynomial5_texture);
				glUniform1i(glGetUniformLocation(program, "polynomial5_texture"), 7);

				glActiveTexture(GL_TEXTURE8);
				glBindTexture(GL_TEXTURE_2D, nl_mask_idx);
				glUniform1i(glGetUniformLocation(program, "nl_output_mask"), 8);
			}


			// parameters
			glUniformMatrix3fv(glGetUniformLocation(program, "R"), 1, GL_FALSE, glm::value_ptr(rotation));
			glUniform3fv(glGetUniformLocation(program, "t"), 1, glm::value_ptr(translation));
			glUniform1f(glGetUniformLocation(program, "w"), w);
			glUniform1f(glGetUniformLocation(program, "h"), h);
			glUniform1f(glGetUniformLocation(program, "n_w"), n_w);
			glUniform1f(glGetUniformLocation(program, "n_h"), n_h);
			glUniform1f(glGetUniformLocation(program, "max_depth"), static_cast<InputView&>(input).get_max_depth());

			auto input_projection_type = ogl_transformer->getInputParameters().getProjectionType();
			auto output_projection_type = ogl_transformer->getVirtualParameters().getProjectionType();
			glUniform1i(glGetUniformLocation(program, "erp_in"), input_projection_type == ProjectionType::equirectangular);
			glUniform1i(glGetUniformLocation(program, "erp_out"), output_projection_type == ProjectionType::equirectangular);

			if (input_projection_type == ProjectionType::perspective) {
				auto f = ogl_transformer->getInputParameters().getFocal();
				auto p = ogl_transformer->getInputParameters().getPrinciplePoint();
				glUniform2fv(glGetUniformLocation(program, "f"), 1, f.val);
				glUniform2fv(glGetUniformLocation(program, "p"), 1, p.val);
			}
			else if (input_projection_type == ProjectionType::equirectangular) {
				auto hor_range = ogl_transformer->getInputParameters().getHorRange();
				auto ver_range = ogl_transformer->getInputParameters().getVerRange();
				auto constexpr radperdeg = 0.01745329252f;
				glUniform1f(glGetUniformLocation(program, "phi0"), radperdeg * hor_range[1]);
				glUniform1f(glGetUniformLocation(program, "theta0"), radperdeg * ver_range[1]);
				glUniform1f(glGetUniformLocation(program, "dphi_du"), -radperdeg * (hor_range[1] - hor_range[0]) / w);
				glUniform1f(glGetUniformLocation(program, "dtheta_dv"), -radperdeg * (ver_range[1] - ver_range[0]) / h);
			}
			else {
				throw std::logic_error("Unknown projection type (with OpenGL)");
			}

			if (output_projection_type == ProjectionType::perspective) {
				auto n_f = ogl_transformer->getVirtualParameters().getFocal();
				auto n_p = ogl_transformer->getVirtualParameters().getPrinciplePoint();
				glUniform2fv(glGetUniformLocation(program, "n_f"), 1, n_f.val);
				glUniform2fv(glGetUniformLocation(program, "n_p"), 1, n_p.val);
			}
			else if (output_projection_type == ProjectionType::equirectangular) {
				auto hor_range = ogl_transformer->getVirtualParameters().getHorRange();
				auto ver_range = ogl_transformer->getVirtualParameters().getVerRange();
				auto constexpr degperrad = 57.295779513f;
				glUniform1f(glGetUniformLocation(program, "u0"), (hor_range[0] + hor_range[1]) / (hor_range[1] - hor_range[0]));
				glUniform1f(glGetUniformLocation(program, "v0"), -(ver_range[0] + ver_range[1]) / (ver_range[1] - ver_range[0]));
				glUniform1f(glGetUniformLocation(program, "du_dphi"), -2.f * degperrad / (hor_range[1] - hor_range[0]));
				glUniform1f(glGetUniformLocation(program, "dv_dtheta"), +2.f * degperrad / (ver_range[1] - ver_range[0]));
			}
			else {
				throw std::logic_error("Unknown projection type (with OpenGL)");
			}

			// end parameters

			glBindVertexArray(vve.VAO);
			//printf("Number of elements %i\n", int(vve.number_of_elements));
			glDrawElements(GL_TRIANGLES, int(vve.number_of_elements), GL_UNSIGNED_INT, nullptr);
			glUseProgram(0);
			glBindVertexArray(0);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			glDisable(GL_DEPTH_TEST);

			glDeleteTextures(GLsizei(1), &image_texture);
			glDeleteTextures(GLsizei(1), &depth_texture);
			if (input.get_displacementMethod() == DisplacementMethod::polynomial) {
				glDeleteTextures(GLsizei(1), &mask_texture);
				glDeleteTextures(GLsizei(1), &polynomial1_texture);
				glDeleteTextures(GLsizei(1), &polynomial2_texture);
				glDeleteTextures(GLsizei(1), &polynomial3_texture);
				glDeleteTextures(GLsizei(1), &polynomial4_texture);
				glDeleteTextures(GLsizei(1), &polynomial5_texture);

				glDeleteTextures(GLsizei(1), &nl_mask_idx);

			}
		}
#endif

		if (!g_with_opengl) {
			cv::Mat2f virtual_uv;
			cv::Mat1f virtual_depth;
			WrappingMethod wrapping_method;

			virtual_uv = pu_transformer->generateImagePos();
			cv::Mat3f input_xyz = pu_transformer->unproject(virtual_uv, inputDepth);
			cv::Mat3f virtual_xyz = affine_transform(input_xyz, R, t);
			virtual_uv = pu_transformer->project(virtual_xyz, virtual_depth, wrapping_method);

			// Resize: rasterize with oversampling
			cv::Size virtual_size = pu_transformer->getVirtualParameters().getSize();
			cv::Size output_size = cv::Size(
				int(0.5f + virtual_size.width * detail::g_rescale),
				int(0.5f + virtual_size.height * detail::g_rescale));
			cv::Mat2f scaled_uv;
			cv::transform(virtual_uv, scaled_uv, cv::Matx22f(float(output_size.width) / virtual_size.width, 0.f, 0.f, float(output_size.height) / virtual_size.height));

			// Rasterization results in a color, depth and quality map
			transform(inputColor, scaled_uv, virtual_depth, output_size, wrapping_method);
		}
	}

	void SynthesizedView::computeCudaMode(View& input)
	{
		assert(m_space_transformer);

		auto R = m_space_transformer->get_rotation();
		auto t = m_space_transformer->get_translation();
		distance_from_origin = cv::max(0.01, cv::norm(t));

		auto const& pu_transformer = static_cast<const GenericTransformer*>(m_space_transformer);
		Parameters inputParams = pu_transformer->getInputParameters();
		Parameters virtualParams = pu_transformer->getVirtualParameters();

		cv::Size size = inputParams.getSize();
		cv::Size input_size = cv::Size(
			static_cast<int>(detail::g_rescale * size.width),
			static_cast<int>(detail::g_rescale * size.height));

		float2* dev_virtual_uv;
		float* dev_virtual_depth;
		WrappingMethod wrapping_method;
		processTransformationCudaMode(inputParams, virtualParams, input_size, input, R, t, dev_virtual_uv, dev_virtual_depth, wrapping_method);

		cv::Size virtual_size = virtualParams.getSize();
		cv::Size output_size = cv::Size(
			int(0.5f + virtual_size.width * detail::g_rescale),
			int(0.5f + virtual_size.height * detail::g_rescale));

		if (virtualParams.getProjectionType() != "Equirectangular" && detail::g_rescale != 1.0f)
		{
			try { scaleUV(dev_virtual_uv, detail::g_rescale, input_size.area()); }
			catch (const std::runtime_error& e) { std::cerr << "CUDA Error (scaleUV) : " << e.what() << std::endl; }
		}

		transformCudaMode(input.getDevColor(), dev_virtual_uv, dev_virtual_depth, input_size, output_size, wrapping_method);
	}

	void SynthesizedView::compute(View& input)
	{
		if (g_with_cuda)
			computeCudaMode(input);
		else
			computeBasicMode(input);
	}

	SynthetisedViewTriangle::SynthetisedViewTriangle() {}

	void SynthetisedViewTriangle::processTransformationCudaMode
	(
		Parameters inputParams, Parameters virtualParams, cv::Size size, View input, cv::Matx33f R, cv::Vec3f t, float2*& dev_virtual_uv, float*& dev_virtual_depth, WrappingMethod& wrapping_method
	)
	{
		try
		{
			bool isERP = virtualParams.getProjectionType() == "Equirectangular";

			if (isERP)
			{
				unprojectERP_projectERP(size,
					input.getDevDepth(),
					dev_virtual_uv,
					dev_virtual_depth,
					inputParams.getHorRange(),
					inputParams.getVerRange(),
					R, t);

				wrapping_method = inputParams.isFullHorRange()
					? WrappingMethod::horizontal
					: WrappingMethod::none;
			}
			else
			{
				unprojectERP_projectPerspective(size,
					input.getDevDepth(),
					dev_virtual_uv,
					dev_virtual_depth,
					inputParams.getHorRange(),
					inputParams.getVerRange(),
					R, t,
					virtualParams.getFocal(),
					virtualParams.getPrinciplePoint());

				wrapping_method = WrappingMethod::none;
			}
		}
		catch (const std::runtime_error& e)
		{
			std::cerr << "CUDA Error (unproj/proj) : " << e.what() << std::endl;
		}
	}

	void SynthetisedViewTriangle::transform
	(
		cv::Mat3f input_color, cv::Mat2f input_positions, cv::Mat1f input_depth, cv::Size output_size, WrappingMethod wrapping_method
	)
	{
		bool wrapHorizontal = (wrapping_method == WrappingMethod::horizontal);

		cv::Mat3f output_color;
		cv::Mat1f output_depth;
		cv::Mat1f validity;

		output_color = detail::transform_trianglesMethod(input_color, input_depth, input_positions, output_size, output_depth, validity, wrapHorizontal);
		assign(output_color, output_depth, validity / output_depth, validity);
	}

	void SynthetisedViewTriangle::transformCudaMode
	(
		float3* input_color, float2* dev_input_positions, float* dev_input_depth, cv::Size input_size, cv::Size output_size, WrappingMethod wrapping_method
	)
	{
		try
		{
			bool wrapHorizontal = (wrapping_method == WrappingMethod::horizontal);

			float3* dev_output_color;
			float* dev_output_depth;
			float* dev_validity;

			transform_trianglesMethod(
				input_color, input_size, dev_input_depth, dev_input_positions, output_size, wrapHorizontal, dev_output_color, dev_output_depth, dev_validity);

			assign(dev_output_color, dev_output_depth, dev_validity, output_size);
		}
		catch (const std::runtime_error& e)
		{
			std::cerr << "CUDA Error (transform) : " << e.what() << std::endl;
		}
	}
}
