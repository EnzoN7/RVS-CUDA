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


# Author of the CUDA version:
#
# Ecole de Technologie Superieure, Montreal, Canada:
#	Enzo Di Maria, https://github.com/EnzoN7/


cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(PROJECT_NAME "RVS")

if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(NVCC_BIN nvcc
        PATHS
            /usr/local/cuda-12.6/bin
            /usr/local/cuda/bin
            /opt/cuda/bin
        NO_DEFAULT_PATH)
    if(NOT NVCC_BIN)
        find_program(NVCC_BIN nvcc)
    endif()
    if(NVCC_BIN)
        set(CMAKE_CUDA_COMPILER ${NVCC_BIN} CACHE FILEPATH "CUDA compiler")
        message(STATUS "nvcc auto-détecté : ${NVCC_BIN}")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${OpenMP_CXX_FLAGS}\"")
endif()

if(MSVC)
	set(variables CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG)

	foreach(variable ${variables})
		if(${variable} MATCHES "/MT")
			string(REGEX REPLACE "/MT" "/MD" ${variable} "${${variable}}")
		endif()

		if(${variable} MATCHES "/MTd")
			string(REGEX REPLACE "/MTd" "/MDd" ${variable} "${${variable}}")
		endif()
	endforeach()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /wd4505")
	set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /PROFILE")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++17")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()

set(PROJECT_SOURCES
	rvs/src/BlendedView.cpp
	rvs/src/Config.cpp
	rvs/src/Parameters.cpp
	rvs/src/JsonParser.cpp
	rvs/src/Pipeline.cpp
	rvs/src/SynthesizedView.cpp
	rvs/src/View.cpp
	rvs/src/image_loading.cpp
	rvs/src/image_writing.cpp
	rvs/src/PoseTraces.cpp
	rvs/src/SpaceTransformer.cpp)

set(PROJECT_HEADERS
	rvs/src/BlendedView.hpp
	rvs/src/Config.hpp
	rvs/src/JsonParser.hpp
	rvs/src/Pipeline.hpp
	rvs/src/SynthesizedView.hpp
	rvs/src/View.hpp
	rvs/src/Parameters.hpp
	rvs/src/image_loading.hpp
	rvs/src/image_writing.hpp
	rvs/src/Projector.hpp
	rvs/src/PoseTraces.hpp
	rvs/src/SpaceTransformer.hpp)
				 						 	 
source_group("Source Files" FILES ${PROJECT_SOURCES} src/view_synthesis.cpp src/test.cpp)
source_group("Header Files" FILES ${PROJECT_HEADERS})

set(CUDA_SOURCES
	rvs_cuda_lib/src/blend_img.cu
	rvs_cuda_lib/src/inpaint_img.cu
	rvs_cuda_lib/src/scale_uv.cu
	rvs_cuda_lib/src/unproject_project.cu
	rvs_cuda_lib/src/import_export_images.cu
	rvs_cuda_lib/src/transform_triangle.cu)
set(CUDA_HEADERS
	rvs_cuda_lib/src/types.cuh
	rvs_cuda_lib/src/blend_img.cuh
	rvs_cuda_lib/src/inpaint_img.cuh
	rvs_cuda_lib/src/scale_uv.cuh
	rvs_cuda_lib/src/unproject_project.cuh
	rvs_cuda_lib/src/import_export_images.cuh
	rvs_cuda_lib/src/transform_triangle.cuh)

source_group("Source Files" FILES ${CUDA_SOURCES})
source_group("Header Files" FILES ${CUDA_HEADERS})

add_library(RVSCudaLib ${CUDA_SOURCES} ${CUDA_HEADERS})

add_library(${PROJECT_NAME}Lib ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_include_directories(${PROJECT_NAME}Lib PUBLIC rvs_cuda_lib/src)

add_executable(${PROJECT_NAME} rvs/src/Application.cpp rvs/src/Application.hpp rvs/src/view_synthesis.cpp)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}Lib ${OpenCV_LIBS} ${CUDA_LIB})
target_link_libraries(${PROJECT_NAME}Lib RVSCudaLib)

file(GLOB_RECURSE FORMAT_SOURCES
    ${CMAKE_SOURCE_DIR}/rvs/src/*.cpp
    ${CMAKE_SOURCE_DIR}/rvs/src/*.hpp
    ${CMAKE_SOURCE_DIR}/rvs_cuda_lib/src/*.cu
    ${CMAKE_SOURCE_DIR}/rvs_cuda_lib/src/*.cuh)

find_program(CLANG_FORMAT_BIN NAMES clang-format-15 clang-format)
if(CLANG_FORMAT_BIN)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_BIN} -i --style=file ${FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatage clang-format (in-place) sur ${PROJECT_NAME}")

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror --style=file ${FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Vérification clang-format sans modification")
else()
    message(STATUS "clang-format introuvable : cibles `format` et `format-check` non créées")
endif()
