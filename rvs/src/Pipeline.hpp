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

#ifndef _PIPELINE_HPP_
#define _PIPELINE_HPP_

#include "View.hpp"
#include "Config.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <future>

namespace rvs 
{
	class BlendedView;
	class SynthesizedView;
	class SpaceTransformer;

	/**
	@file Pipeline.hpp
	\brief The file containing the pipeline functions
	*/

	/**
	\brief The pipeline of view synthesis

	The pipeline executes the following steps:
		- Parsing the configuration file (see Parser)
		- Loading the input reference views (see InputView, and load_images());
		- View synthesis of the target view once for each input reference view (see SynthesizedView);
		- Blending all the SynthesizedView together by assigning a per-pixel quality to each synthesized view (see BlendedView);
		- Inpainting to fill the remaining holes (see inpaint());
		- Writing the output (see write_color()).
	*/
	class Pipeline
	{
	public:
		/**
		\brief Constructor
		*/
		Pipeline();

		/**
		\brief Execution of the view synthesis

		Execution consists in parsing, image loading, view computation (warping, blending, inpainting), view writing
		*/
		int execute();

	protected:
		/**
		\brief Interface to provide access to the configuration
		*/
		virtual Config const& getConfig() const = 0;

		/**
		\brief Interface for loading a source view. Implemented by Application
		*/
		virtual std::shared_ptr<View> loadInputView(int inputFrame, int inputView, Parameters const& parameters) = 0;

		/**
		\brief Does the derived class want to save color?
		*/
		virtual bool wantColor();

		/**
		\brief Does the derived class want to save masked color?
		*/
		virtual bool wantMaskedColor();

		/**
		\brief Does the derived class want to save the validity mask itself?
		*/
		virtual bool wantMask();

		/**
		\brief does the derived class want to save the depth map?
		*/
		virtual bool wantDepth();

		/**
		\brief does the derived class want to save the masked depth map?
		*/
		virtual bool wantMaskedDepth();

		/**
		\brief Interface for saving a regular (inpainted) synthesis result. Implemented by Application
		*/
		virtual void saveColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters);
		virtual void saveColor(float3* color, int virtualFrame, int virtualView, Parameters const& parameters, cv::Size size);

		/**
		\brief Interface for saving a masked synthesis result. Implemented by Application
		*/
		virtual void saveMaskedColor(cv::Mat3f color, int virtualFrame, int virtualView, Parameters const& parameters);

		/**
		\brief Interface for saving the validity mask. Implemented by Application
		*/
		virtual void saveMask(cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const& parameters);

		/**
		\brief Interface for saving the synthesized depth map. Implemented by Application
		*/
		virtual void saveDepth(cv::Mat1f depth, int virtualFrame, int virtualView, Parameters const& parameters);

		/**
		\brief Interface for saving the synthesized depth map. Implemented by Application
		*/
		virtual void saveMaskedDepth(cv::Mat1f depth, cv::Mat1b mask, int virtualFrame, int virtualView, Parameters const& parameters);

		/**
		\brief Interface for making intermediate result available for pruning or analysis

		Pipeline calls this function after synthesizing a single input view
		*/
		virtual void onIntermediateSynthesisResult(int inputFrame, int inputView, int virtualFrame, int virtualView, SynthesizedView const& synthesizedView);

		/**
		\brief Interface for making intermediate result available for pruning or analysis

		Pipeline calls this function after synthesizing a single input view
		*/
		virtual void onIntermediateBlendingResult(int inputFrame, int inputView, int virtualFrame, int virtualView, BlendedView const& blendedView);

		/**
		\brief Interface for making intermediate result available for pruning or analysis

		Pipeline calls this function after synthesizing a single input view
		With OpenGL the result is available through opengl::RFBO::getInstance()
		Otherwise the result is available through the blendedView parameter
		*/
		virtual void onFinalBlendingResult(int inputFrame, int virtualFrame, int virtualView, BlendedView const& blendedView);

		/**
		\brief Create a view blender to blend the warped input views to one virtual view
		*/
		virtual std::unique_ptr<BlendedView> createBlender(int virtualView);

		/**
		\brief Create a space transformer to reproject the input view to the virtual view
		*/
		virtual std::unique_ptr<SpaceTransformer> createSpaceTransformer(int virtualView);

		/**
		\brief Create a view synthesizer to warp one input view to one virtual view
		*/
		virtual std::unique_ptr<SynthesizedView> createSynthesizer(int inputView, int virtualView);

	private:
		/**
		\brief Computes one frame of a virtual view

		Executes the view view computation (warping, blending, inpainting) and writing.
		@param inputFrame Input frame number of the frame to compute
		@param virtualFrame Virtual (output) frame number of the frame to compute
		@param virtualView Index of the virtual view to compute
		*/
		void computeView(int inputFrame, int virtualFrame, int virtualView, std::shared_ptr<View> inputImage, std::future<std::shared_ptr<View>>& nextImageFuture, std::future<void>& saveColorFuture);
	};
}

#endif
