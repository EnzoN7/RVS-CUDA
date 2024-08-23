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

#include "helpersGL.hpp"
#include "RFBO.hpp"
#include "Shader.hpp"

#include <string>

#include "gl_core_4.5.hpp"

// OpenGL Utilities

#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>

#if _WIN32
#include <gl/gl.h>
#else
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

namespace rvs
{
	namespace opengl
	{
		GLuint cvMat2glTexture(const cv::Mat& mat)
		{
			// https://stackoverflow.com/questions/16809833/opencv-image-loading-for-opengl-texture
			// REMARK: The image is flipped horizontally as OpenCV stores data in the opposite direction of OpenGL

			double min, max;
			cv::minMaxLoc(mat, &min, &max);
			cv::Mat img = mat;
			/*
			//why ?
			if (max > 1)
				img /= max;*/

				//mat.convertTo(mat, CV_8UC3);

				//use fast 4-byte alignment (default anyway) if possible
			glPixelStorei(GL_UNPACK_ALIGNMENT, (img.step & 3) ? 1 : 4); //printf("%i %llu ", (mat.step & 3), mat.step / mat.elemSize());

																		//set length of one complete row in data (doesn't need to equal image.cols)
			glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(img.step / img.elemSize()));

			GLenum internalformat = GL_RGB32F;
			if (img.channels() == 4) internalformat = GL_RGBA32F;
			if (img.channels() == 3) internalformat = GL_RGB;
			if (img.channels() == 2) internalformat = GL_RG;
			if (img.channels() == 1) internalformat = GL_R32F;

			GLenum externalformat = GL_BGR;
			if (img.channels() == 1) externalformat = GL_RED; // GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT32F, GL_R32F NOT WORKING!
			if (img.channels() == 4) externalformat = GL_RGBA; // GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT32F, GL_R32F NOT WORKING!

			GLuint texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);


			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			try {
				//internalformat = internals[k];
				glTexImage2D(GL_TEXTURE_2D,
					/* level */				0,
					/* internalFormat */	internalformat,
					/* width */				img.cols,
					/* height */			img.rows,
					/* border */			0,
					/* format */			externalformat,
					/* type */				GL_FLOAT,
					/* *data */				img.ptr());

				//glGenerateMipmap(GL_TEXTURE_2D);
			}
			catch (std::exception & e)
			{
				printf("%s.\n", e.what());
			}

			return texture;
		}

		// END OpenGL Utilities


		// We need to declare this global variable to make it available everywhere
		context_NO_WRITE_H context_NO_WRITE;

#if  /*!defined NDEBUG &&*/  WITH_RENDERDOC
		RENDERDOC_API_1_1_2 *rdoc_api = nullptr;
#endif

#if _WIN32
		// Inspired by https://gist.github.com/nickrolfe/1127313ed1dbf80254b614a721b3ee9c

		typedef HGLRC WINAPI wglCreateContextAttribsARB_type(HDC hdc, HGLRC hShareContext,
			const int *attribList);
		wglCreateContextAttribsARB_type *wglCreateContextAttribsARB;

		// See https://www.opengl.org/registry/specs/ARB/wgl_create_context.txt for all values
#define WGL_CONTEXT_MAJOR_VERSION_ARB             0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB             0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB              0x9126

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001

		typedef BOOL WINAPI wglChoosePixelFormatARB_type(HDC hdc, const int *piAttribIList,
			const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
		wglChoosePixelFormatARB_type *wglChoosePixelFormatARB;

		// See https://www.opengl.org/registry/specs/ARB/wgl_pixel_format.txt for all values
#define WGL_DRAW_TO_WINDOW_ARB                    0x2001
#define WGL_ACCELERATION_ARB                      0x2003
#define WGL_SUPPORT_OPENGL_ARB                    0x2010
#define WGL_DOUBLE_BUFFER_ARB                     0x2011
#define WGL_PIXEL_TYPE_ARB                        0x2013
#define WGL_COLOR_BITS_ARB                        0x2014
#define WGL_DEPTH_BITS_ARB                        0x2022
#define WGL_STENCIL_BITS_ARB                      0x2023

#define WGL_FULL_ACCELERATION_ARB                 0x2027
#define WGL_TYPE_RGBA_ARB                         0x202B

		static void
			init_opengl_extensions(void)
		{
			// Before we can load extensions, we need a dummy OpenGL context, created using a dummy window.
			// We use a dummy window because you can only set the pixel format for a window once. For the
			// real window, we want to use wglChoosePixelFormatARB (so we can potentially specify options
			// that aren't available in PIXELFORMATDESCRIPTOR), but we can't load and use that before we
			// have a context.
			WNDCLASSA window_class = {
				/* UINT      style = */         CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
				/* WNDPROC   lpfnWndProc = */   DefWindowProcA,
				/* int       cbClsExtra = */	0,
				/* int       cbWndExtra = */	0,
				/* HINSTANCE hInstance = */		GetModuleHandle(0),
				/* HICON     hIcon = */			NULL,
				/* HCURSOR   hCursor = */		NULL,
				/* HBRUSH    hbrBackground = */ NULL,
				/* LPCTSTR   lpszMenuName = */	NULL,
				/* LPCTSTR   lpszClassName = */ "Dummy_WGL_windows"
			};

			if (!RegisterClassA(&window_class)) {
				printf("Failed to register dummy OpenGL window.\n");
			}

			HWND dummy_window = CreateWindowExA(
				/* _In_     DWORD     dwExStyle, */     0,
				/* _In_opt_ LPCTSTR   lpClassName, */   window_class.lpszClassName,
				/* _In_opt_ LPCTSTR   lpWindowName, */	"Dummy_OpenGL_Window",
				/* _In_     DWORD     dwStyle, */		0,
				/* _In_     int       x, */				CW_USEDEFAULT,
				/* _In_     int       y, */				CW_USEDEFAULT,
				/* _In_     int       nWidth, */		CW_USEDEFAULT,
				/* _In_     int       nHeight, */		CW_USEDEFAULT,
				/* _In_opt_ HWND      hWndParent, */	0,
				/* _In_opt_ HMENU     hMenu, */			0,
				/* _In_opt_ HINSTANCE hInstance, */		window_class.hInstance,
				/* _In_opt_ LPVOID    lpParam */		0
			);

			if (!dummy_window) {
				printf("Failed to create dummy OpenGL window.\n");
			}

			HDC dummy_dc = GetDC(dummy_window);

			PIXELFORMATDESCRIPTOR pfd;
			ZeroMemory(&pfd, sizeof(pfd));
			pfd.nSize = sizeof(pfd);
			pfd.nVersion = 1;
			pfd.iPixelType = PFD_TYPE_RGBA;
			pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
			pfd.cColorBits = 32;
			pfd.cAlphaBits = 8;
			pfd.iLayerType = PFD_MAIN_PLANE;
			pfd.cDepthBits = 24;
			pfd.cStencilBits = 8;

			int pixel_format = ChoosePixelFormat(dummy_dc, &pfd);
			if (!pixel_format) {
				printf("Failed to find a suitable pixel format.\n");
			}
			if (!SetPixelFormat(dummy_dc, pixel_format, &pfd)) {
				printf("Failed to set the pixel format.\n");
			}

			HGLRC dummy_context = wglCreateContext(dummy_dc);
			if (!dummy_context) {
				printf("Failed to create a dummy OpenGL rendering context.\n");
			}

			if (!wglMakeCurrent(dummy_dc, dummy_context)) {
				printf("Failed to activate dummy OpenGL rendering context.\n");
			}

			wglCreateContextAttribsARB = (wglCreateContextAttribsARB_type*)wglGetProcAddress(
				"wglCreateContextAttribsARB");
			wglChoosePixelFormatARB = (wglChoosePixelFormatARB_type*)wglGetProcAddress(
				"wglChoosePixelFormatARB");

			wglMakeCurrent(dummy_dc, 0);
			wglDeleteContext(dummy_context);
			ReleaseDC(dummy_window, dummy_dc);
			DestroyWindow(dummy_window);
		}


		static HGLRC init_opengl(HDC real_dc)
		{
			init_opengl_extensions();

			// Now we can choose a pixel format the modern way, using wglChoosePixelFormatARB.
			int pixel_format_attribs[] = {
				WGL_DRAW_TO_WINDOW_ARB,     GL_TRUE,
				WGL_SUPPORT_OPENGL_ARB,     GL_TRUE,
				WGL_DOUBLE_BUFFER_ARB,      GL_TRUE,
				WGL_ACCELERATION_ARB,       WGL_FULL_ACCELERATION_ARB,
				WGL_PIXEL_TYPE_ARB,         WGL_TYPE_RGBA_ARB,
				WGL_COLOR_BITS_ARB,         32,
				WGL_DEPTH_BITS_ARB,         24,
				WGL_STENCIL_BITS_ARB,       8,
				0
			};

			int pixel_format;
			UINT num_formats;
			wglChoosePixelFormatARB(real_dc, pixel_format_attribs, 0, 1, &pixel_format, &num_formats);
			if (!num_formats) {
				printf("Failed to choose the OpenGL 4.5 pixel format.\n");
			}

			PIXELFORMATDESCRIPTOR pfd;
			DescribePixelFormat(real_dc, pixel_format, sizeof(pfd), &pfd);
			if (!SetPixelFormat(real_dc, pixel_format, &pfd)) {
				printf("Failed to set the OpenGL 4.5 pixel format.\n");
			}

			// Specify that we want to create an OpenGL 3.3 core profile context
			int gl33_attribs[] = {
				WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
				WGL_CONTEXT_MINOR_VERSION_ARB, 3,
				WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
				0,
			};

			HGLRC gl45_context = wglCreateContextAttribsARB(real_dc, 0, gl33_attribs);
			if (!gl45_context) {
				printf("Failed to create OpenGL 4.5 context.\n");
			}

			if (!wglMakeCurrent(real_dc, gl45_context)) {
				printf("Failed to activate OpenGL 4.5 rendering context.\n");
			}

			return gl45_context;
		}

		static LRESULT CALLBACK
			window_callback(HWND window, UINT msg, WPARAM wparam, LPARAM lparam)
		{
			LRESULT result = 0;

			switch (msg) {
			case WM_CLOSE:
			case WM_DESTROY:
				PostQuitMessage(0);
				break;
			default:
				result = DefWindowProcA(window, msg, wparam, lparam);
				break;
			}

			return result;
		}

		static HWND
			create_window(HINSTANCE inst)
		{
			WNDCLASSA window_class;
			ZeroMemory(&window_class, sizeof(window_class));
			window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
			window_class.lpfnWndProc = window_callback;
			window_class.hInstance = inst;
			window_class.hCursor = LoadCursor(0, IDC_ARROW);
			window_class.hbrBackground = 0;
			window_class.lpszClassName = "WGL_fdjhsklf";

			if (!RegisterClassA(&window_class)) {
				printf("Failed to register window.\n");
			}

			// Specify a desired width and height, then adjust the rect so the window's client area will be
			// that size.
			RECT rect;
			ZeroMemory(&rect, sizeof(rect));
			rect.right = 100;
			rect.bottom = 100;

			DWORD window_style = WS_OVERLAPPEDWINDOW;
			AdjustWindowRect(&rect, window_style, false);

			HWND window = nullptr;
			window = CreateWindowExA(
				/* _In_     DWORD     dwExStyle, */     0,
				/* _In_opt_ LPCTSTR   lpClassName, */   window_class.lpszClassName,
				/* _In_opt_ LPCTSTR   lpWindowName, */	"OpenGL_Window",
				/* _In_     DWORD     dwStyle, */		0,
				/* _In_     int       x, */				CW_USEDEFAULT,
				/* _In_     int       y, */				CW_USEDEFAULT,
				/* _In_     int       nWidth, */		CW_USEDEFAULT, /* rect.right - rect.left, */
				/* _In_     int       nHeight, */		CW_USEDEFAULT, /* rect.bottom - rect.top, */
				/* _In_opt_ HWND      hWndParent, */	0,
				/* _In_opt_ HMENU     hMenu, */			0,
				/* _In_opt_ HINSTANCE hInstance, */		window_class.hInstance,
				/* _In_opt_ LPVOID    lpParam */		0
			);

			if (!window) {
				printf("Failed to create window.\n");
			}

			return window;
		}
#else // linux
		// Inspired by http://apoorvaj.io/creating-a-modern-opengl-context.html
		typedef GLXContext(*glXCreateContextAttribsARBProc) (Display*, GLXFBConfig, GLXContext, Bool, const int*);

		void create_opengl_context()
		{
			/* Create_display_and_window
			-------------------------
			Skip if you already have a display and window */
			context_NO_WRITE.disp = XOpenDisplay(0);
			context_NO_WRITE.win = XCreateSimpleWindow(context_NO_WRITE.disp, DefaultRootWindow(context_NO_WRITE.disp),
				10, 10,   /* x, y */
				800, 600, /* width, height */
				0, 0,     /* border_width, border */
				0);       /* background */

			/* Create_the_modern_OpenGL_context
			 -------------------------------- */
			static int visual_attribs[] = {
				GLX_RENDER_TYPE, GLX_RGBA_BIT,
				GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
				GLX_DOUBLEBUFFER, true,
				GLX_RED_SIZE, 1,
				GLX_GREEN_SIZE, 1,
				GLX_BLUE_SIZE, 1,
				None
			};

			int num_fbc = 0;
			GLXFBConfig *fbc = glXChooseFBConfig(context_NO_WRITE.disp,
				DefaultScreen(context_NO_WRITE.disp),
				visual_attribs, &num_fbc);
			if (!fbc) {
				printf("glXChooseFBConfig() failed\n");
				exit(1);
			}

			/* Create old OpenGL context to get correct function pointer for
			glXCreateContextAttribsARB() */
			XVisualInfo *vi = glXGetVisualFromFBConfig(context_NO_WRITE.disp, fbc[0]);
			GLXContext ctx_old = glXCreateContext(context_NO_WRITE.disp, vi, 0, GL_TRUE);
			glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
			glXCreateContextAttribsARB =
				(glXCreateContextAttribsARBProc)
				glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
			/* Destroy old context */
			glXMakeCurrent(context_NO_WRITE.disp, 0, 0);
			glXDestroyContext(context_NO_WRITE.disp, ctx_old);
			if (!glXCreateContextAttribsARB) {
				printf("glXCreateContextAttribsARB() not found\n");
				exit(1);
			}

			/* Set desired minimum OpenGL version */
			static int context_attribs[] = {
				GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
				GLX_CONTEXT_MINOR_VERSION_ARB, 2,
				None
			};
			/* Create modern OpenGL context */
			context_NO_WRITE.ctx = glXCreateContextAttribsARB(context_NO_WRITE.disp, fbc[0], NULL, true, context_attribs);
			if (!context_NO_WRITE.ctx) {
				printf("Failed to create OpenGL context. Exiting.\n");
				exit(1);
			}
		}

		void show_window(Display * disp, Window & win, GLXContext & ctx)
		{
			XMapWindow(disp, win);
			glXMakeCurrent(disp, win, ctx);

			int major = 0, minor = 0;
			glGetIntegerv(GL_MAJOR_VERSION, &major);
			glGetIntegerv(GL_MINOR_VERSION, &minor);
			printf("OpenGL context created.\nVersion %d.%d\nVendor %s\nRenderer %s\n",
				major, minor,
				glGetString(GL_VENDOR),
				glGetString(GL_RENDERER));
		}
#endif

		void context_init() {
			if (context_NO_WRITE.initialized)
				return;

#if  /*!defined NDEBUG &&*/  WITH_RENDERDOC
			// At init, on windows
#if _WIN32
			HMODULE mod = GetModuleHandleA("renderdoc.dll");
			if (mod)
			{
				const pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
				const int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
				assert(ret == 1);
			}
			else
			{
				printf("Unable to load RenderDoc module.\n");
			}
#endif
#endif

#if _WIN32
			context_NO_WRITE.fakewindow = create_window(GetModuleHandle(nullptr));

			context_NO_WRITE.gldc = GetDC(context_NO_WRITE.fakewindow);
			context_NO_WRITE.glrc = init_opengl(context_NO_WRITE.gldc);
#else // linux
			create_opengl_context();
#endif
			if (ogl_LoadFunctions() == ogl_LOAD_FAILED)
			{
				printf("Failed to load OpenGL Functions.\n");
			}

			// Show a window:
			// Windows:
			//ShowWindow(context_NO_WRITE.fakewindow, 1);
			//UpdateWindow(context_NO_WRITE.fakewindow); // in the main loop
			// Linux:
			//show_window(context_NO_WRITE.disp, context_NO_WRITE.win, context_NO_WRITE.ctx);
			context_NO_WRITE.initialized = true;
		}

		void setGLContext()
		{
#if _WIN32
			wglMakeCurrent(context_NO_WRITE.gldc, context_NO_WRITE.glrc);
#else // linux
			//glXMakeCurrent(context_NO_WRITE.disp, context_NO_WRITE.win, context_NO_WRITE.ctx);
			glXMakeContextCurrent(context_NO_WRITE.disp, context_NO_WRITE.win, context_NO_WRITE.win, context_NO_WRITE.ctx);
#endif
		}


		void rd_start_capture_frame() {
#if /*!defined NDEBUG &&*/ WITH_RENDERDOC
			if (rdoc_api && rdoc_api->IsTargetControlConnected()) {
				//rdoc_api->TriggerCapture();
				rdoc_api->StartFrameCapture(NULL, NULL);
			}
#endif
		}

		void rd_end_capture_frame() {
#if /*!defined NDEBUG &&*/ WITH_RENDERDOC
			static int frame_number = 0;
			if (rdoc_api && rdoc_api->IsTargetControlConnected()) {
				//rdoc_api->TriggerCapture();
				rdoc_api->EndFrameCapture(NULL, NULL);
				printf("RENDERDOC - Frame %i Captured.\n", frame_number);
				frame_number++;
			}
#endif
		}
	}
}
