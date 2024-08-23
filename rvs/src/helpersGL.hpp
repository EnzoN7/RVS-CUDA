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

#ifndef _HELPERS_GL_HPP_
#define _HELPERS_GL_HPP_

#if !WITH_OPENGL
#error "This header requires WITH_OPENGL"
#endif

#include "gl_core_4.5.hpp"
#include "Config.hpp"
#include <opencv2/core/mat.hpp>

// C4201 in glm: nonstandard extension used: nameless struct/union
#if _MSC_VER >= 1900
#pragma warning(disable : 4201)
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#if _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#endif

#if WITH_RENDERDOC
#include <renderdoc_app.h>
extern RENDERDOC_API_1_1_2 *rdoc_api;
#endif

namespace rvs
{
	namespace opengl
	{
		// OpenGL Utilities

		template<int rows, int cols>
		bool fromCV2GLM(const cv::Mat& cvmat, glm::mat<cols, rows, float, glm::packed_highp>* glmmat) {
			assert(cvmat.cols == cols && cvmat.rows == rows && cvmat.type() == CV_32FC1);

			memcpy(glm::value_ptr(*glmmat), cvmat.data, rows*cols * sizeof(float));
			*glmmat = glm::transpose(*glmmat);
			return true;
		}

		template<int rows, int cols>
		bool fromGLM2CV(const glm::mat<cols, rows, float, glm::packed_highp>& glmmat, cv::Mat* cvmat) {
			if (cvmat->cols != cols || cvmat->rows != rows) {
				(*cvmat) = cv::Mat(rows, cols, 4, CV_32F);
			}
			memcpy(cvmat->data, glm::value_ptr(glmmat), rows*cols * sizeof(float));
			*cvmat = cvmat->t();
			return true;
		}

		GLuint cvMat2glTexture(const cv::Mat& mat);

		// END OpenGL Utilities

		struct VAO_VBO_EBO {
			GLuint VAO;
			GLuint VBO;
			GLuint EBO;
			size_t number_of_elements;

			VAO_VBO_EBO(cv::Size size) 
			{
				cv::Mat1f empty_value = cv::Mat1f::zeros(size);
				//use fast 4-byte alignment (default anyway) if possible
				glPixelStorei(GL_UNPACK_ALIGNMENT, (empty_value.step & 3) ? 1 : 4);

				//set length of one complete row in data (doesn't need to equal image.cols)
				glPixelStorei(GL_UNPACK_ROW_LENGTH, GLint(empty_value.step / empty_value.elemSize()));

				glGenVertexArrays(1, &VAO);
				glBindVertexArray(VAO);

				glGenBuffers(1, &VBO);
				glBindBuffer(GL_ARRAY_BUFFER, VBO);

				glBufferData(GL_ARRAY_BUFFER, empty_value.rows*empty_value.cols*empty_value.elemSize(), empty_value.ptr(), GL_STATIC_DRAW);

				generate_picture_EBO(size, number_of_elements);

				GLuint num_components_per_vertex = 1; // depth
				glVertexAttribPointer(0, num_components_per_vertex, GL_FLOAT, GL_FALSE,
					1 * sizeof(GLfloat),
					(GLvoid*)0);
				glEnableVertexAttribArray(0);


				glBindVertexArray(0);

			}

			~VAO_VBO_EBO() {
				glDeleteBuffers(1, &EBO);
				glDeleteBuffers(1, &VBO);
				glDeleteVertexArrays(1, &VAO);
			}

#define INDEX_E(x,y,W) ((y)*(W) + (x))
			void generate_picture_EBO(const cv::Size &s, size_t &elements_number)
			{
				const size_t W = s.width;
				const size_t H = s.height;

				elements_number = 3 * 2 * (H - 1)*(W - 1);
				std::vector<GLuint> indices;
				indices.resize(elements_number);

				size_t offset = 0;
				for (size_t y = 0; y < H - 1; ++y)
				{
					for (size_t x = 0; x < W - 1; ++x)
					{
						indices[3 * INDEX_E(x, y, (W - 1)) + 0] = GLuint(INDEX_E(x, y, W));
						indices[3 * INDEX_E(x, y, (W - 1)) + 1] = GLuint(INDEX_E(x + 1, y, W));
						indices[3 * INDEX_E(x, y, (W - 1)) + 2] = GLuint(INDEX_E(x, y + 1, W));
						offset += 3;
					}
				}

				for (size_t y = 1; y < H; ++y)
				{
					for (size_t x = 0; x < W - 1; ++x)
					{
						indices[offset + 3 * INDEX_E(x, y - 1, (W - 1)) + 0] = GLuint(INDEX_E(x, y, W));
						indices[offset + 3 * INDEX_E(x, y - 1, (W - 1)) + 1] = GLuint(INDEX_E(x + 1, y - 1, W));
						indices[offset + 3 * INDEX_E(x, y - 1, (W - 1)) + 2] = GLuint(INDEX_E(x + 1, y, W));
					}
				}

				glGenBuffers(1, &EBO);
				// Bind index buffer to corresponding target
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				// ititialize index buffer, allocate memory, fill it with data
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements_number * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

				printf("Real number of elements %i\n", int(elements_number));
			}

		};

		// NEVER WRITE ON IT.
		struct context_NO_WRITE_H
		{
#if _WIN32
			HWND fakewindow = nullptr;
			HDC gldc = nullptr;
			HGLRC glrc = nullptr;
#else
			Display * disp = nullptr;
			Window win = 0;
			GLXContext ctx = 0;
#endif
			bool initialized = false;
		};

		extern context_NO_WRITE_H context_NO_WRITE;

		void context_init();
		void setGLContext();

		void rd_start_capture_frame();
		void rd_end_capture_frame();
	}
}

#endif
