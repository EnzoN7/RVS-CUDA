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
set(CMAKE_CUDA_ARCHITECTURES 86)

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
			src/gl_core_4.5.cpp
			src/helpersGL.cpp
			src/RFBO.cpp
			src/Shader.cpp)

		set(OPENGL_HEADERS
			src/gl_core_4.5.hpp
			src/helpersGL.hpp
			src/RFBO.hpp
			src/Shader.hpp)
		
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
	src/BlendedView.cpp
	src/PolynomialDepth.cpp
	src/Config.cpp
	src/Parameters.cpp
	src/JsonParser.cpp
	src/Pipeline.cpp
	src/SynthesizedView.cpp
	src/View.cpp
	src/blending.cpp
	src/image_loading.cpp
	src/image_writing.cpp
	src/inpainting.cpp
	src/transform.cpp
	src/Unprojector.cpp
	src/Projector.cpp
	src/PerspectiveUnprojector.cpp
	src/PerspectiveProjector.cpp
	src/EquirectangularUnprojector.cpp
	src/EquirectangularProjector.cpp
	src/PoseTraces.cpp
	src/SpaceTransformer.cpp)

set(PROJECT_HEADERS
	src/BlendedView.hpp
	src/PolynomialDepth.hpp
	src/Config.hpp
	src/JsonParser.hpp
	src/Pipeline.hpp
	src/SynthesizedView.hpp
	src/View.hpp
	src/blending.hpp
	src/Parameters.hpp
	src/image_loading.hpp
	src/image_writing.hpp
	src/inpainting.hpp
	src/transform.hpp
	src/IntegralImage2D.h
	src/Unprojector.hpp
	src/Projector.hpp
	src/PerspectiveUnprojector.hpp
	src/PerspectiveProjector.hpp
	src/EquirectangularUnprojector.hpp
	src/EquirectangularProjector.hpp
	src/PoseTraces.hpp
	src/SpaceTransformer.hpp)

set(CONFIGURATION_FILES
	config_files/example_config_file.json
	config_files/ClassroomVideo.json
	config_files/TechnicolorMuseum.json
	config_files/TechnicolorHijack.json
	config_files/ULB_Unicorn.json
	config_files/_integration_tests/Unicorn_Same_View.json
	config_files/_integration_tests/Unicorn_Triangles_MultiSpectral.json
	config_files/_integration_tests/Unicorn_Triangles_Simple.json
	config_files/_integration_tests/Unicorn_Triangles_Simple_OpenGL.json
	config_files/_integration_tests/ClassroomVideo-v0_to_v0.json
	config_files/_integration_tests/ClassroomVideo-v7v8_to_v0.json
	config_files/_integration_tests/ClassroomVideo-v7v8_to_v0_270deg.json
	config_files/_integration_tests/ClassroomVideo-v7v8_to_v0_OpenGL.json
	config_files/_integration_tests/TechnicolorMuseum-v0_to_v0.json
	config_files/_integration_tests/TechnicolorMuseum-v0v2v13v17v19_to_v1.json
	config_files/_integration_tests/TechnicolorMuseum-v0v2v13v17v19_to_v1_OpenGL.json
	config_files/_integration_tests/TechnicolorMuseum-v5_to_v5.json
	config_files/_integration_tests/TechnicolorMuseum-v5_to_v6.json
	config_files/_integration_tests/TechnicolorMuseum-PoseTrace.json
	config_files/_integration_tests/TechnicolorHijack-BlendByMax.json
	config_files/_integration_tests/TechnicolorHijack-v1v4_to_v9.json
	config_files/_integration_tests/TechnicolorHijack-v1v4_to_v9_OpenGL.json)
				 						 	 
source_group("Source Files" FILES ${PROJECT_SOURCES} src/view_synthesis.cpp src/test.cpp)
source_group("Header Files" FILES ${PROJECT_HEADERS})
source_group("Configurations" FILES ${CONFIGURATION_FILES})
if (WITH_OPENGL)
	source_group("OpenGL Sources" FILES ${OPENGL_SOURCES})
	source_group("OpenGL Headers" FILES ${OPENGL_HEADERS})
endif()

set(CUDA_SOURCES
	../RVSCuda/src/blend_img.cu
	../RVSCuda/src/inpaint_img.cu
	../RVSCuda/src/scale_uv.cu
	../RVSCuda/src/unproject_project.cu
	../RVSCuda/src/process_input_frame.cu
	../RVSCuda/src/transform_triangle.cu)
set(CUDA_HEADERS
	../RVSCuda/src/blend_img.cuh
	../RVSCuda/src/inpaint_img.cuh
	../RVSCuda/src/scale_uv.cuh
	../RVSCuda/src/unproject_project.cuh
	../RVSCuda/src/process_input_frame.cuh
	../RVSCuda/src/transform_triangle.cuh)

source_group("Source Files" FILES ${CUDA_SOURCES})
source_group("Header Files" FILES ${CUDA_HEADERS})

add_library(RVSCuda ${CUDA_SOURCES} ${CUDA_HEADERS})
add_library(${PROJECT_NAME}Lib ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${OPENGL_SOURCES} ${OPENGL_HEADERS})

add_executable(${PROJECT_NAME} src/Application.cpp src/Analyzer.cpp src/Application.hpp src/Analyzer.hpp src/view_synthesis.cpp)
add_executable(${PROJECT_NAME}UnitTest src/unit_test.cpp)
add_executable(${PROJECT_NAME}IntegrationTest src/Application.cpp src/integration_test.cpp ${CONFIGURATION_FILES})

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}Lib ${OpenCV_LIBS} ${CUDA_LIB} ${EASYPROFILER_LIB} ${OPENGL_LIB})
target_link_libraries(${PROJECT_NAME}UnitTest ${PROJECT_NAME}Lib ${OpenCV_LIBS} ${CUDA_LIB} ${EASYPROFILER_LIB} ${OPENGL_LIB})
target_link_libraries(${PROJECT_NAME}IntegrationTest ${PROJECT_NAME}Lib ${OpenCV_LIBS} ${CUDA_LIB} ${EASYPROFILER_LIB} ${OPENGL_LIB})
target_link_libraries(${PROJECT_NAME}Lib RVSCuda)

target_include_directories(${PROJECT_NAME}Lib PUBLIC ../RVSCuda/src)

enable_testing()
add_test(NAME UnitTest${PROJECT_NAME} COMMAND ${PROJECT_NAME}UnitTest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME IntegrationTest${PROJECT_NAME} COMMAND ${PROJECT_NAME}IntegrationTest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
