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

// NOTE: Integration tests require the following files to be downloaded and unpacked
//       (or symlinked) in the root of the source tree:
//
//   *  Plane_B'_Kinect.zip from ULB and on the MPEG content server
//   *  Plane_B'_Kinect-10bit.tar.gz from Philips to be uploaded

#include "yaffut.hpp"

#include "Application.hpp"

#include <array>
#include <fstream>

#include <opencv2/imgproc.hpp>

#if WITH_OPENGL
#include "helpersGL.hpp"
#endif

const std::string sourcePath = "";

namespace testing
{
	template<typename T> std::array<cv::Mat_<T>, 3> readYUV420(char const *filepath, cv::Size size)
	{
		// Open the file stream
		std::ifstream stream;
		stream.exceptions(std::ios::badbit | std::ios::failbit);
		stream.open(filepath, std::ios::binary);

		// Test file length is correct
		stream.seekg(0, std::ios::end);
		auto filesize = static_cast<std::size_t>(stream.tellg());
		stream.seekg(0, std::ios::beg);
		auto framesize = size.area() * 3 / 2 * sizeof(T);
		std::cout << "size == " << size << ", sizeof(T) == " << sizeof(T) << ", filesize == " << filesize << ", framesize == " << framesize << std::endl;
		YAFFUT_CHECK(filesize && filesize % framesize == 0);

		// Allocate planes
		cv::Mat_<T> Y(size);
		cv::Mat_<T> Cb(size / 2);
		cv::Mat_<T> Cr(size / 2);

		// Read the frame
		stream.read(reinterpret_cast<char*>(Y.data), size.area() * sizeof(T));
		stream.read(reinterpret_cast<char*>(Cb.data), size.area() / 4 * sizeof(T));
		stream.read(reinterpret_cast<char*>(Cr.data), size.area() / 4 * sizeof(T));

		return{ Y, Cb, Cr };
	}

	// Compare two YUV420 frames and check PSNR is below a threshold	
	// Two regions are taken because errors are large at the image borders
	template<typename T> void compareWithReferenceView(
		std::array<cv::Mat_<T>, 3> const& actual,
		std::array<cv::Mat_<T>, 3> const& reference,
		int bits, double threshold0, double threshold1)
	{
		YAFFUT_CHECK(actual[0].size() == reference[0].size());
		YAFFUT_CHECK(actual[1].size() == reference[1].size());
		YAFFUT_CHECK(actual[2].size() == reference[2].size());

		auto size = actual[0].size();
		double psnr[2];

		for (int i_region = 0; i_region != 2; ++i_region) {
			auto sum_sq_error = 0.;
			std::size_t num_values = 0;
			auto num_planes = 3;

			for (int i_plane = 0; i_plane != num_planes; ++i_plane) {
				auto plane_size = i_plane ? size / 2 : size;

				cv::Range cols, rows;
				switch (i_region) {
				case 0: 
					cols = cv::Range(0, plane_size.width);
					rows = cv::Range(0, plane_size.height);
					break;

				case 1:
					cols = cv::Range(plane_size.width / 3, plane_size.width * 2 / 3);
					rows = cv::Range(plane_size.height / 3, plane_size.height * 2 / 3);
					break;
				}

				auto sqrt_sum_sq_error = norm(
					actual[i_plane](rows, cols),
					reference[i_plane](rows, cols));

				num_values += cols.size() * rows.size();
				sum_sq_error += sqrt_sum_sq_error * sqrt_sum_sq_error;
			}

			auto mean_sq_error = sum_sq_error / num_values;
			auto max_level = (1 << bits) - 1;
			psnr[i_region] = 10. * std::log10(max_level * max_level / mean_sq_error);
		}

		std::cout << "PSNRs: " << psnr[0] << ", " << psnr[1] << std::endl;
		YAFFUT_CHECK(psnr[0] > threshold0);
		YAFFUT_CHECK(psnr[1] > threshold1);

		if ((psnr[0] > threshold0 + 0.1 && threshold0 < 100.) ||
			(psnr[1] > threshold1 + 0.1 && threshold1 < 100.)) {
			std::clog << "WARNING: thresholds can be increased\n";
		}
	}

	template<typename T> void compareWithReferenceView(
		char const *filepath_actual,
		char const *filepath_reference,
		cv::Size size, int bits,
		double threshold0, double threshold1)
	{
		std::cout << "Comparing \"" << filepath_actual << "\" " << size << " with \"" << filepath_reference << "\" " << size << std::endl;
		auto actual = readYUV420<T>(filepath_actual, size);
		
		std::string filepath_reference_mod = sourcePath.empty()
			? filepath_reference
			: sourcePath + "/" + filepath_reference;
        
        auto reference = readYUV420<T>(filepath_reference_mod.c_str(), size);
		compareWithReferenceView(actual, reference, bits, threshold0, threshold1);
	}
}

namespace rvs
{
	extern bool g_verbose;
}

FUNC(ULB_Unicorn_Example)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/example_config_file.json", sourcePath);
	p.execute();
	// No reference
}

FUNC(ULB_Unicorn_Triangles_Simple)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/Unicorn_Triangles_Simple.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint8_t>(
		"030003250438_from_030003070370_030003430506_Triangles_Simple.yuv",
		"Plane_B'/Plane_B'_Texture/Kinect_z0300y0325x0438.yuv",
		cv::Size(1920, 1080), 8, 21.08, 26.12); // VC15 + OpenCV 3.4.1: 21.1318, 26.1742

#if WITH_OPENGL
	rvs::g_with_opengl = true;
	rvs::opengl::context_init();
	rvs::Application pgl("./config_files/_integration_tests/Unicorn_Triangles_Simple_OpenGL.json");
	pgl.execute();

	// OpenGL vs reference
	testing::compareWithReferenceView<std::uint8_t>(
		"Plane_B'/Plane_B'_Texture/Kinect_z0300y0325x0438.yuv",
		"030003250438_from_030003070370_030003430506_Triangles_Simple_OpenGL.yuv",
		cv::Size(1920, 1080), 8, 21.08, 26.08); // VC15 + OpenCV 3.4.1 + Quadro K420: 21.136, 26.1372

	// No OpenGL vs OpenGL
	testing::compareWithReferenceView<std::uint8_t>(
		"030003250438_from_030003070370_030003430506_Triangles_Simple.yuv",
		"030003250438_from_030003070370_030003430506_Triangles_Simple_OpenGL.yuv",
		cv::Size(1920, 1080), 8, 34.84, 35.87); // VC15 + OpenCV 3.4.1 + Quadro K420: 34.8416, 35.9293
#endif
}

FUNC(ULB_Unicorn_Triangles_MultiSpectral)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/Unicorn_Triangles_MultiSpectral.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint8_t>(
		"030003250438_from_030003070370_030003430506_Triangles_Multispectral.yuv",
		"Plane_B'/Plane_B'_Texture/Kinect_z0300y0325x0438.yuv",
		cv::Size(1920, 1080), 8, 21.06, 25.89); // VC15 + OpenCV 3.4.1: 21.1138, 25.9494
}

FUNC(ULB_Unicorn_Same_View)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/Unicorn_Same_View.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint8_t>(
		"030003250506_from_030003250506_Same_View.yuv",
		"Plane_B'/Plane_B'_Texture/Kinect_z0300y0325x0506.yuv",
		cv::Size(1920, 1080), 8, 19.79, 34.03); // VC15 + OpenCV 3.4.1: 19.8489, 34.0849
}

FUNC(ClassroomVideo_v0_to_v0)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/ClassroomVideo-v0_to_v0.json", sourcePath);
	p.execute();

	// No OpenGL vs reference: texture
	testing::compareWithReferenceView<std::uint16_t>(
		"v0vs_4096_2048_420_10b.yuv",
		"ClassroomVideo/v0_4096_2048_420_10b.yuv",
		cv::Size(4096, 2048), 10, 70.00, 78.52); // VC15 + OpenCV 3.4.1: 70.034, 78.572

	// No OpenGL vs reference: depth maps
	testing::compareWithReferenceView<std::uint16_t>(
		"v0vs_4096_2048_0_8_1000_0_420_10b.yuv",
		"ClassroomVideo/v0_4096_2048_0_8_1000_0_420_10b.yuv",
		cv::Size(4096, 2048), 10, 100., 100.); // VC15 + OpenCV 3.4.1: inf, inf
}

FUNC(ClassroomVideo_v7v8_to_v0)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/ClassroomVideo-v7v8_to_v0.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"v0vs_from_v7v8_4096_2048_420_10b.yuv",
		"ClassroomVideo/v0_4096_2048_420_10b.yuv",
		cv::Size(4096, 2048), 10, 35.37, 36.27); // VC15 + OpenCV 3.4.1: 35.3846, 36.2868

#if WITH_OPENGL
	rvs::g_with_opengl = true;
	rvs::opengl::context_init();
	rvs::Application pGL("./config_files/_integration_tests/ClassroomVideo-v7v8_to_v0_OpenGL.json");
	pGL.execute();

	// OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"ClassroomVideo/v0_4096_2048_420_10b.yuv",
		"v0vs_from_v7v8_4096_2048_420_10b_OpenGL.yuv",
		cv::Size(4096, 2048), 10, 35.40, 35.26); // VC15 + OpenCV 3.4.1 + Quadro K420: 35.4502, 35.3126

	// No OpenGL vs. OpenGL
	testing::compareWithReferenceView<std::uint16_t>(
		"v0vs_from_v7v8_4096_2048_420_10b.yuv",
		"v0vs_from_v7v8_4096_2048_420_10b_OpenGL.yuv",
		cv::Size(4096, 2048), 10, 42.65, 42.59); // VC15 + OpenCV 3.4.1 + Quadro K420: 42.7035, 42.6449
#endif
}

FUNC(ClassroomVideo_v7v8_to_v0_270deg)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/ClassroomVideo-v7v8_to_v0_270deg.json", sourcePath);
	p.execute();

	auto actual = testing::readYUV420<std::uint16_t>("v0_270deg_from_v7v8_2304_1536_420_10b.yuv", cv::Size(2304, 1536));

    std::string filepath_reference = sourcePath + "ClassroomVideo/v0_4096_2048_420_10b.yuv";
	auto reference = testing::readYUV420<std::uint16_t>( filepath_reference.c_str(), cv::Size(4096, 2048));
	cv::resize(reference[0], reference[0], cv::Size(3072, 1536));
	cv::resize(reference[1], reference[1], cv::Size(1536, 768));
	cv::resize(reference[2], reference[2], cv::Size(1536, 768));
	reference[0] = reference[0].colRange(384, 2688);
	reference[1] = reference[1].colRange(192, 1344);
	reference[2] = reference[2].colRange(192, 1344);

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(actual, reference, 10, 37.44, 37.86); // VC15 + OpenCV 3.4.1: 37.4546, 37.8746
}

FUNC(TechnicolorHijack_v1v4_to_v9)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorHijack-v1v4_to_v9.json", sourcePath);
	p.execute();

	// No OpenGL vs reference: texture
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorHijack_v9vs_from_v1v4_4096_4096_420_10b.yuv",
		"TechnicolorHijack/v9_4096_4096_420_10b.yuv",
		cv::Size(4096, 4096), 10, 43.50, 35.71); // VC15 + OpenCV 3.4.1:  43.5123, 35.7278
	
	// No OpenGL vs reference: depth
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorHijack_v9vs_from_v1v4_4096_4096_0_5_25_0_420_10b.yuv",
		"TechnicolorHijack/v9_4096_4096_0_5_25_0_420_10b.yuv",
		cv::Size(4096, 4096), 10, 23.12, 39.03); // VC15 + OpenCV 3.4.1: 23.1774, 39.0874

#if WITH_OPENGL
	rvs::g_with_opengl = true;
	rvs::opengl::context_init();
	rvs::Application pGL("./config_files/_integration_tests/TechnicolorHijack-v1v4_to_v9_OpenGL.json");
	pGL.execute();

	// OpenGL vs. reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorHijack_v9vs_from_v1v4_4096_4096_420_10b_OpenGL.yuv",
		"TechnicolorHijack/v9_4096_4096_420_10b.yuv",
		cv::Size(4096, 4096), 10, 41.81, 35.62); // VC15 + OpenCV 3.4.1 + Quadro K420: 41.8636, 35.6741

	// No OpenGL vs. OpenGL
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorHijack_v9vs_from_v1v4_4096_4096_420_10b.yuv",
		"TechnicolorHijack_v9vs_from_v1v4_4096_4096_420_10b_OpenGL.yuv",
		cv::Size(4096, 4096), 10, 44.38, 39.04); // VC15 + OpenCV 3.4.1 + Quadro K420: 44.4393, 39.0989
#endif
}

FUNC(TechnicolorHijack_BlendByMax)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorHijack-BlendByMax.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorHijack_BlendByMax.yuv",
		"TechnicolorHijack/v9_4096_4096_420_10b.yuv",
		cv::Size(4096, 4096), 10, 42.86, 35.00); // VC15 + OpenCV 3.4.1:  42.8731, 35.0181
}

FUNC(TechnicolorMuseum_v0v2v13v17v19_to_v1)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-v0v2v13v17v19_to_v1.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v1vs_from_v0v2v13v17v19_2048_2048_420_10b.yuv",
		"TechnicolorMuseum/v1_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 31.88, 38.42); // VC15 + OpenCV 3.4.1:   31.8979, 38.4389

#if WITH_OPENGL
	rvs::g_with_opengl = true;
	rvs::opengl::context_init();
	rvs::Application pGL("./config_files/_integration_tests/TechnicolorMuseum-v0v2v13v17v19_to_v1_OpenGL.json");
	pGL.execute();

	// OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v1vs_from_v0v2v13v17v19_2048_2048_420_10b_OpenGL.yuv",
		"TechnicolorMuseum/v1_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 27.16, 38.21); // VC15 + OpenCV 3.4.1 + Quadro K420: 27.2167, 38.2646
												 
	// No OpenGL vs OpenGL
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v1vs_from_v0v2v13v17v19_2048_2048_420_10b.yuv",
		"TechnicolorMuseum_v1vs_from_v0v2v13v17v19_2048_2048_420_10b_OpenGL.yuv",
		cv::Size(2048, 2048), 10, 28.51, 44.21); // VC15 + OpenCV 3.4.1 + Quadro K420: 28.5609, 44.2661
#endif
}

FUNC(TechnicolorMuseum_v0_to_v0)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-v0_to_v0.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v0vs_from_v0_2048_2048_420_10b.yuv",
		"TechnicolorMuseum/v0_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 64.65, 73.57); // VC15 + OpenCV 3.4.1: 64.6653, 73.5891
}

FUNC(TechnicolorMuseum_v5_to_v5)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-v5_to_v5.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v5vs_from_v5_2048_2048_420_10b.yuv",
		"TechnicolorMuseum/v5_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 58.37, 73.77); // VC15 + OpenCV 3.4.1:   58.3812, 73.7879
}

FUNC(TechnicolorMuseum_v5_to_v6)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-v5_to_v6.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v6vs_from_v5_2048_2048_420_10b.yuv",
		"TechnicolorMuseum/v6_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 21.83, 26.50); // VC15 + OpenCV 3.4.1:  21.8415, 26.5167
}

FUNC(TechnicolorMuseum_PoseTrace)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-PoseTrace.json", sourcePath);
	p.execute();

	// No OpenGL vs reference
	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_PoseTrace.yuv",
		"TechnicolorMuseum/v6_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 21.83, 26.50); // VC15 + OpenCV 3.4.1:   21.8415, 26.5167
}

FUNC(TechnicolorMuseum_translucency)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-translucency.json", sourcePath);
	p.execute();

	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v8vs_from_v3v8_2048x2048_yuv420p10le.yuv",
		"TechnicolorMuseum/v8_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 43.23, 42.31); // GCC 4.9.2: 43.2881, 42.3646

	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v5vs_from_v3v8_2048x2048_yuv420p10le.yuv",
		"TechnicolorMuseum/v5_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 29.29, 31.07); // GCC 4.9.2: 29.3411, 31.1295
}

FUNC(TechnicolorMuseum_translucency_inverse)
{
	rvs::g_with_opengl = false;
	rvs::Application p("./config_files/_integration_tests/TechnicolorMuseum-translucency-inverse.json", sourcePath);
	p.execute();

	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v8vs_from_v8v3_2048x2048_yuv420p10le.yuv",
		"TechnicolorMuseum/v8_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 43.23, 42.31); // GCC 4.9.2: 43.2881, 42.3646

	testing::compareWithReferenceView<std::uint16_t>(
		"TechnicolorMuseum_v5vs_from_v8v3_2048x2048_yuv420p10le.yuv",
		"TechnicolorMuseum/v5_2048_2048_420_10b.yuv",
		cv::Size(2048, 2048), 10, 29.29, 31.07); // GCC 4.9.2: 29.3411, 31.1295 
}

int main(int argc, const char* argv[])
{
	rvs::g_verbose = true;
	cv::setBreakOnError(true);
	return yaffut::main(argc, argv);
}
