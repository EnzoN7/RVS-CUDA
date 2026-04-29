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

#if WITH_OPENGL
#include "helpersGL.hpp"
#endif

#include "Analyzer.hpp"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace rvs
{
	extern bool g_verbose;
}

int main(int argc, char* argv[])
{
	try
	{
		rvs::g_verbose = true;
		bool with_analyzer = false;
		std::string filename;

		for (int i = 1; i < argc; ++i) {
			if (strcmp(argv[i], "--opengl") == 0) {
				rvs::g_with_opengl = true;
			}
			else if (strcmp(argv[i], "--cuda") == 0) {
				rvs::g_with_cuda = true;
			}
			else if (strcmp(argv[i], "--analyzer") == 0) {
				with_analyzer = true;
			}
			else if (strcmp(argv[i], "--help") == 0) {
				filename.clear();
				break;
			}
			else if (filename.empty()) {
				filename = argv[i];
			}
			else {
				throw std::runtime_error("Too many parameters (try --help)");
			}
		}
		
		std::cout
			<< " - -------------------------------------------------------------------------------------- -\n"
			<< "|    Reference View Synthesizer (RVS), branch: v4.0-dev                                    |\n"
			<< "|                                                                                          |\n"
			<< "|    MPEG2018/N18068 Reference View Synthesizer (RVS) manual                               |\n"
			<< " - -------------------------------------------------------------------------------------- -" << std::endl;

		if (filename.empty()) {
			std::cout
				<< "\n"
				<< " - -------------------------------------------------------------------------------------- -\n"
				<< "|    Original authors:                                                                     |\n"
				<< "|                                                                                          |\n"
				<< "|    Universite Libre de Bruxelles, Brussels, Belgium:                                     |\n"
				<< "|      Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be                              |\n"
				<< "|      Daniele Bonatto, Daniele.Bonatto@ulb.ac.be                                          |\n"
				<< "|      Arnaud Schenkel, arnaud.schenkel@ulb.ac.be                                          |\n"
				<< "|                                                                                          |\n"
				<< "|    Koninklijke Philips N.V., Eindhoven, The Netherlands:                                 |\n"
				<< "|      Bart Kroon, bart.kroon@philips.com                                                  |\n"
				<< "|      Bart Sonneveldt, bart.sonneveldt@philips.com                                        |\n"
				<< " - -------------------------------------------------------------------------------------- -\n\n";

			throw std::runtime_error("Usage: RVS CONFIGURATION_FILE [--noopengl] [--analyzer]");
		}

#if WITH_OPENGL
		if (rvs::g_with_opengl) {
			rvs::opengl::context_init();
		}
#else
		rvs::g_with_opengl = false;
#endif
		
		std::unique_ptr<rvs::Application> application;

		if (with_analyzer) {
			application.reset(new rvs::Analyzer(filename));
		}
		else {
			application.reset(new rvs::Application(filename));
		}

		cudaSetDevice(0);
		int device;
		cudaGetDevice(&device);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);

		std::cout << "==============================================================================================" << std::endl;
		std::cout << "		GPU : " << prop.name << std::endl;
		std::cout << "		ID  : " << device << std::endl;
		std::cout << "==============================================================================================" << std::endl;

		auto startTime = clock();

		int number_of_output_frames = application->execute();
    
		auto executeTime = double(clock() - startTime) / CLOCKS_PER_SEC;
        
		std::cout  
			<< std::endl
			<< "Total time:         " << std::fixed << std::setprecision(3) << executeTime
			<< " sec."
			<< std::endl;

		std::cout
			<< std::endl
			<< "Av. speed:          " << std::fixed << std::setprecision(3) << number_of_output_frames / executeTime
			<< " fps.\n"
			<< "Av. time per frame: " << std::fixed << std::setprecision(3) << executeTime / number_of_output_frames
			<< " sec."
			<< std::endl;

		return 0;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
}

