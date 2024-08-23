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

#include "RFBO.hpp"

namespace rvs
{
	namespace opengl
	{
		RFBO * RFBO::_singleton = nullptr;

		void RFBO::toggle()
		{
			validate();
			clear_working_buffers();
			value = (value + 1) % 2;
		}

		void RFBO::clear_working_buffers()
		{
			validate();

			// OpenGL 4.4+
			glClearTexImage(swap_image[value], 0, GL_RGB, GL_FLOAT, 0);
			glClearTexImage(swap_quality[value], 0, GL_RED, GL_FLOAT, 0);


			glClearTexImage(image, 0, GL_RGB, GL_FLOAT, 0);
			glClearTexImage(depth, 0, GL_RED, GL_FLOAT, 0);
			glClearTexImage(quality_triangle, 0, GL_RED, GL_FLOAT, 0);
		}

		void RFBO::clear_buffers()
		{
			validate();

			// OpenGL 4.4+
			glClearTexImage(swap_image[0], 0, GL_RGB, GL_FLOAT, 0);
			glClearTexImage(swap_quality[0], 0, GL_RED, GL_FLOAT, 0);
			glClearTexImage(swap_image[1], 0, GL_RGB, GL_FLOAT, 0);
			glClearTexImage(swap_quality[1], 0, GL_RED, GL_FLOAT, 0);


			glClearTexImage(image, 0, GL_RGB, GL_FLOAT, 0);
			glClearTexImage(depth, 0, GL_RED, GL_FLOAT, 0);
			glClearTexImage(quality_triangle, 0, GL_RED, GL_FLOAT, 0);
		}

		GLuint RFBO::create_texture_buffer(cv::Size size, GLenum attachement_type, GLenum internalformat)
		{
			validate();
			GLenum intf = GL_RED;
			if (internalformat == GL_RGB32F)
				intf = GL_RGB;

			GLuint texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(GL_TEXTURE_2D, 0, internalformat, size.width, size.height, 0, intf, GL_UNSIGNED_BYTE, 0);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

			glFramebufferTexture(GL_FRAMEBUFFER, attachement_type, texture, 0);

			return texture;
		}
		void RFBO::delete_texture_buffer(GLuint texture)
		{
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &texture);
		}

		void RFBO::create_fbo(cv::Size size)
		{
			setGLContext();
			glViewport(0, 0, size.width, size.height);

			glGenFramebuffers(1, &ID);
			glBindFramebuffer(GL_FRAMEBUFFER, ID);
			initialized = true;

			image = create_texture_buffer(size, GL_COLOR_ATTACHMENT0, GL_RGB32F);
			depth = create_texture_buffer(size, GL_COLOR_ATTACHMENT1, GL_R32F);
			quality_triangle = create_texture_buffer(size, GL_COLOR_ATTACHMENT2, GL_R32F);

			swap_image[0] = create_texture_buffer(size, GL_COLOR_ATTACHMENT3, GL_RGB32F);
			swap_quality[0] = create_texture_buffer(size, GL_COLOR_ATTACHMENT4, GL_R32F);

			swap_image[1] = create_texture_buffer(size, GL_COLOR_ATTACHMENT5, GL_RGB32F);
			swap_quality[1] = create_texture_buffer(size, GL_COLOR_ATTACHMENT6, GL_R32F);

			glGenRenderbuffers(1, &depth_stencil);
			glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, size.width, size.height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_stencil);

			GLuint attachments[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 ,GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6 };
			glDrawBuffers(7, attachments);

			if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			{
				initialized = false;
				printf("FATAL ERROR WHILE CREATING THE FRAMEBUFFER. %u\n", glGetError());
				exit(1);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		}

		void RFBO::free()
		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glDeleteFramebuffers(1, &ID);
			initialized = false;

			delete_texture_buffer(image);
			delete_texture_buffer(depth);
			delete_texture_buffer(quality_triangle);

			delete_texture_buffer(swap_image[0]);
			delete_texture_buffer(swap_quality[0]);

			delete_texture_buffer(swap_image[1]);
			delete_texture_buffer(swap_quality[1]);

			glBindRenderbuffer(GL_RENDERBUFFER, 0);
			glDeleteRenderbuffers(1, &depth_stencil);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 0, 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);


			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		}
	}
}
