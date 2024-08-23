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
  Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be  Daniele Bonatto, Daniele.Bonatto@ulb.ac.be  Arnaud Schenkel, arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
  Bart Kroon, bart.kroon@philips.com
  Bart Sonneveldt, bart.sonneveldt@philips.com
*/

#ifndef _RFBO_HPP_
#define _RFBO_HPP_

#if !WITH_OPENGL
#error "This header requires WITH_OPENGL"
#endif

#include "helpersGL.hpp"

namespace rvs
{
	namespace opengl
	{
		class RFBO
		{
		private:
			RFBO() {}
			~RFBO()
			{
				// TODO delete OpenGL Buffers	
				delete _singleton;
				initialized = false;
			}

		public:
			static RFBO * getInstance()
			{
				if (_singleton == nullptr)
					_singleton = new RFBO();
				return _singleton;
			}

			void init(cv::Size size)
			{
				if (!initialized)
					create_fbo(size);
				initialized = true;
			}

		public:

			void toggle();

			void clear_working_buffers();

			void clear_buffers();

			void free();

		private:
			GLuint create_texture_buffer(cv::Size size, GLenum attachement_type, GLenum internalformat);
			void delete_texture_buffer(GLuint texture);

			void create_fbo(cv::Size size);

		public:
			GLuint ID = 0;

			// Input image
			GLuint image = 0;
			GLuint depth = 0;
			GLuint quality_triangle = 0;

			// Swapping images blending
			GLuint swap_image[2] = { 0, 0 };
			GLuint swap_quality[2] = { 0, 0 };

			GLuint depth_stencil = 0;

			GLint value = 0;

		private:
			void validate(void) const
			{
				if (!initialized)
					throw std::logic_error("Framebuffer Object not initialized.");
			}


		private:
			static RFBO * _singleton;
			bool initialized = false;
		};
	}
}

#endif
