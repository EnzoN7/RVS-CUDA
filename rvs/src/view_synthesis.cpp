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

/*
Author of the CUDA version:

Ecole de Technologie Superieure, Montreal, Canada:
	Enzo Di Maria, https://github.com/EnzoN7/
*/

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "types.cuh"
#include "Application.hpp"

namespace rvs
{
	extern bool g_verbose;
}

int main(int argc, char* argv[])
{
	try
	{
		rvs::g_verbose = true;
		std::string filename;

		bool mode = false;

		for (int i = 1; i < argc; ++i)
		{
			if (strcmp(argv[i], "--help") == 0)
			{
				filename.clear();
				break;
			}
			else if (filename.empty())
			{
				filename = argv[i];
			}
			else if (strcmp(argv[i], "--fp16") == 0 && !mode)
			{
				rvs::detail::g_FP32 = false;
				mode = true;
			}
			else if (strcmp(argv[i], "--fp64") == 0 && !mode)
			{
				rvs::detail::g_FP32 = false;
				rvs::detail::g_FP64 = true;
				mode = true;
			}
			else
			{
				throw std::runtime_error("Too many parameters (try --help)");
			}
		}
		
		std::cout
			<< std::endl
			<< "    ----------------------------------------------------\n"
			<< "   |                                                    |\n"
			<< "   |  Reference View Synthesizer GPU - CUDA (RVS-CUDA)  |\n"
			<< "   |                                                    |\n"
			<< "    ----------------------------------------------------" << std::endl;

		if (filename.empty())
		{
			throw std::runtime_error("Usage: RVS CONFIGURATION_FILE");
		}

		cudaSetDevice(0);
		int device;
		cudaGetDevice(&device);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);

		if (rvs::detail::g_FP32)
		{
			std::unique_ptr<rvs::Application<float2, float3, float>> application;
			application.reset(new rvs::Application<float2, float3, float>(filename));

			std::cout
				<< "  FP32 Version" << std::endl << std::endl
				<< "  * GPU        " << prop.name << std::endl
				<< "  * ID         " << device << std::endl;

			application->execute();
		}
		else if (rvs::detail::g_FP64)
		{
			std::unique_ptr<rvs::Application<double2, double3, double>> application;
			application.reset(new rvs::Application<double2, double3, double>(filename));

			std::cout
				<< "  FP64 Version" << std::endl << std::endl
				<< "  * GPU        " << prop.name << std::endl
				<< "  * ID         " << device << std::endl;

			application->execute();
		}
		else
		{
			std::unique_ptr<rvs::Application<half2, half3, half>> application;
			application.reset(new rvs::Application<half2, half3, half>(filename));

			std::cout
				<< "  FP16 Version" << std::endl << std::endl
				<< "  * GPU        " << prop.name << std::endl
				<< "  * ID         " << device << std::endl;

			application->execute();
		}

		return 0;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
}

