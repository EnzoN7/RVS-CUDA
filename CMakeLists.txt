# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
# 
# Copyright (c) 2010-2018, ITU/ISO/IEC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


# Original authors:
# 
# Universite Libre de Bruxelles, Brussels, Belgium:
#   Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be
#   Daniele Bonatto, Daniele.Bonatto@ulb.ac.be
#   Arnaud Schenkel, arnaud.schenkel@ulb.ac.be
# 
# Koninklijke Philips N.V., Eindhoven, The Netherlands:
#   Bart Kroon, bart.kroon@philips.com
#   Bart Sonneveldt, bart.sonneveldt@philips.com


cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
set(PROJECT_NAME "RVS")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/cmake/easy_profiler/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/cmake/glm/")

find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# force dynamic runtime libraries for msvc builds
if(MSVC)
  set(variables 
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_MINSIZEREL
  )
  foreach(variable ${variables})
    if(${variable} MATCHES "/MT")
      string(REGEX REPLACE "/MT" "/MD" ${variable} "${${variable}}")
    endif()
    if(${variable} MATCHES "/MTd")
      string(REGEX REPLACE "/MTd" "/MDd" ${variable} "${${variable}}")
    endif()
  endforeach()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /PROFILE")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()


option(WITH_EASY_PROFILER "Build the easy_profiler" OFF)

set(EASYPROFILER_LIB "")
if(WITH_EASY_PROFILER)
	find_package(easy_profiler QUIET)
	# D:/LIB/INSTALL/easy_profiler/lib/cmake/easy_profiler
	if(easy_profiler_FOUND)
		set(EASYPROFILER_LIB easy_profiler)
		add_definitions(-DWITH_EASY_PROFILER)
		option(EASY_PROFILER_TO_FILE "Save the profile to a file" OFF)
		if (EASY_PROFILER_TO_FILE)
			add_definitions(-DWITH_EASY_PROFILER_TO_FILE)
		endif()
	endif()
endif()

option(WITH_OPENGL "Build the opengl version" ON)

set(OPENGL_SOURCES "")
set(OPENGL_HEADERS "")
set(OPENGL_LIB "")
if(WITH_OPENGL)
	set(OpenGL_GL_PREFERENCE "GLVND")
	find_package(OpenGL REQUIRED)
	find_package(glm REQUIRED)
	if(OPENGL_FOUND AND glm_FOUND)
		include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})
		include_directories(SYSTEM ${GLM_INCLUDE_DIRS})
		set(OPENGL_LIB ${OPENGL_LIBRARY})

		set(OPENGL_SOURCES
			rvs/src/gl_core_4.5.cpp
			rvs/src/helpersGL.cpp
			rvs/src/RFBO.cpp
			rvs/src/Shader.cpp)

		set(OPENGL_HEADERS
			rvs/src/gl_core_4.5.hpp
			rvs/src/helpersGL.hpp
			rvs/src/RFBO.hpp
			rvs/src/Shader.hpp)
		
		add_definitions(-DWITH_OPENGL)

		if (UNIX)
			find_package(X11 QUIET)
			if (X11_FOUND)
				include_directories(SYSTEM ${X11_INCLUDE_DIR})
				set(OPENGL_LIB ${OPENGL_LIB} ${X11_LIBRARIES})
			endif()
		endif(UNIX)

		find_path(RENDERDOC_INCLUDE_DIR NAMES "renderdoc_app.h" PATHS "C:/Program Files/RenderDoc")
		if(EXISTS ${RENDERDOC_INCLUDE_DIR})
			add_definitions(-DWITH_RENDERDOC)
			include_directories(${RENDERDOC_INCLUDE_DIR})
		else()
			add_definitions(-DWITH_RENDERDOC=false)
		endif()
	endif()
else()
	add_definitions(-DWITH_OPENGL=false)
endif()

set(PROJECT_SOURCES
	rvs/src/BlendedView.cpp
	rvs/src/PolynomialDepth.cpp
	rvs/src/Config.cpp
	rvs/src/Parameters.cpp
	rvs/src/JsonParser.cpp
	rvs/src/Pipeline.cpp
	rvs/src/SynthesizedView.cpp
	rvs/src/View.cpp
	rvs/src/blending.cpp
	rvs/src/image_loading.cpp
	rvs/src/image_writing.cpp
	rvs/src/inpainting.cpp
	rvs/src/transform.cpp
	rvs/src/Unprojector.cpp
	rvs/src/Projector.cpp
	rvs/src/PerspectiveUnprojector.cpp
	rvs/src/PerspectiveProjector.cpp
	rvs/src/EquirectangularUnprojector.cpp
	rvs/src/EquirectangularProjector.cpp
	rvs/src/PoseTraces.cpp
	rvs/src/SpaceTransformer.cpp)

set(PROJECT_HEADERS
	rvs/src/BlendedView.hpp
	rvs/src/PolynomialDepth.hpp
	rvs/src/Config.hpp
	rvs/src/JsonParser.hpp
	rvs/src/Pipeline.hpp
	rvs/src/SynthesizedView.hpp
	rvs/src/View.hpp
	rvs/src/blending.hpp
	rvs/src/Parameters.hpp
	rvs/src/image_loading.hpp
	rvs/src/image_writing.hpp
	rvs/src/inpainting.hpp
	rvs/src/transform.hpp
	rvs/src/IntegralImage2D.h
	rvs/src/Unprojector.hpp
	rvs/src/Projector.hpp
	rvs/src/PerspectiveUnprojector.hpp
	rvs/src/PerspectiveProjector.hpp
	rvs/src/EquirectangularUnprojector.hpp
	rvs/src/EquirectangularProjector.hpp
	rvs/src/PoseTraces.hpp
	rvs/src/SpaceTransformer.hpp)
				 						 	 
source_group("Source Files" FILES ${PROJECT_SOURCES} src/view_synthesis.cpp src/test.cpp)
source_group("Header Files" FILES ${PROJECT_HEADERS})
if (WITH_OPENGL)
	source_group("OpenGL Sources" FILES ${OPENGL_SOURCES})
	source_group("OpenGL Headers" FILES ${OPENGL_HEADERS})
endif()

set(CUDA_SOURCES
	rvs_cuda_lib/src/blend_img.cu
	rvs_cuda_lib/src/inpaint_img.cu
	rvs_cuda_lib/src/scale_uv.cu
	rvs_cuda_lib/src/unproject_project.cu
	rvs_cuda_lib/src/process_input_frame.cu
	rvs_cuda_lib/src/transform_triangle.cu)
set(CUDA_HEADERS
	rvs_cuda_lib/src/blend_img.cuh
	rvs_cuda_lib/src/inpaint_img.cuh
	rvs_cuda_lib/src/scale_uv.cuh
	rvs_cuda_lib/src/unproject_project.cuh
	rvs_cuda_lib/src/process_input_frame.cuh
	rvs_cuda_lib/src/transform_triangle.cuh)

source_group("Source Files" FILES ${CUDA_SOURCES})
source_group("Header Files" FILES ${CUDA_HEADERS})

add_library(RVSCudaLib ${CUDA_SOURCES} ${CUDA_HEADERS})
add_library(${PROJECT_NAME}Lib ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${OPENGL_SOURCES} ${OPENGL_HEADERS})

add_executable(${PROJECT_NAME} rvs/src/Application.cpp rvs/src/Analyzer.cpp rvs/src/Application.hpp rvs/src/Analyzer.hpp rvs/src/view_synthesis.cpp)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}Lib ${OpenCV_LIBS} ${CUDA_LIB} ${EASYPROFILER_LIB} ${OPENGL_LIB})
target_link_libraries(${PROJECT_NAME}Lib RVSCudaLib)

target_include_directories(${PROJECT_NAME}Lib PUBLIC rvs_cuda_lib/src)
