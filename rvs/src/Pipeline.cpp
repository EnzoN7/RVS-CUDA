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

#include "Pipeline.hpp"
#include "BlendedView.hpp"
#include "SynthesizedView.hpp"
#include "inpainting.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <future>

#if WITH_OPENGL
#include "helpersGL.hpp"
#include "RFBO.hpp"
#endif

namespace rvs
{
	Pipeline::Pipeline()
	{
#ifndef NDEBUG
		cv::setBreakOnError(true);
#endif
	}

	int Pipeline::execute()
	{
		std::shared_ptr<View> inputImage;
		std::future<std::shared_ptr<View>> nextImageFuture;
		std::future<void> saveColorFuture;

		for (auto virtualFrame = 0; virtualFrame < getConfig().number_of_output_frames; ++virtualFrame)
		{
			auto inputFrame = getConfig().start_frame + virtualFrame;

			if (getConfig().number_of_output_frames > 1)
				std::cout << std::string(5, '=') << " FRAME " << inputFrame << ' ' << std::string(80, '=') << std::endl;

			assert(getConfig().VirtualCameraNames.size() == 1);

			computeView(inputFrame, virtualFrame, 0, inputImage, nextImageFuture, saveColorFuture);
		}
		return getConfig().number_of_output_frames;
	}

	bool Pipeline::wantColor()
	{
		return false;
	}

	bool Pipeline::wantMaskedColor()
	{
		return false;
	}

	bool Pipeline::wantMask()
	{
		return false;
	}

	bool Pipeline::wantDepth()
	{
		return false;
	}

	bool Pipeline::wantMaskedDepth()
	{
		return false;
	}

	void Pipeline::saveColor(cv::Mat3f, int, int, Parameters const&)
	{
		throw std::logic_error(std::string(__func__) + "not implemented");
	}

	void Pipeline::saveColor(float3*, int, int, Parameters const&, cv::Size)
	{
		throw std::logic_error(std::string(__func__) + "not implemented");
	}

	void Pipeline::saveMaskedColor(cv::Mat3f, int, int, Parameters const&)
	{
		throw std::logic_error(std::string(__func__) + "not implemented");
	}

	void Pipeline::saveMask(cv::Mat1b, int, int, Parameters const&)
	{
		throw std::logic_error(std::string(__func__) + " not implemented");
	}

	void Pipeline::saveDepth(cv::Mat1f, int, int, Parameters const&)
	{
		throw std::logic_error(std::string(__func__) + " not implemented");
	}

	void Pipeline::saveMaskedDepth(cv::Mat1f, cv::Mat1b, int, int, Parameters const&)
	{
		throw std::logic_error(std::string(__func__) + " not implemented");
	}

	void Pipeline::onIntermediateSynthesisResult(int, int, int, int, SynthesizedView const&) {}

	void Pipeline::onIntermediateBlendingResult(int, int, int, int, BlendedView const&) {}

	void Pipeline::onFinalBlendingResult(int, int, int, BlendedView const&) {}

	auto getExtendedIndex(int outputFrameIndex, int numberOfInputFrames) {

		if (numberOfInputFrames <= 0) {
			throw std::runtime_error("Cannot extend frame index with zero input frames");
		}
		const auto frameGroupIndex = outputFrameIndex / numberOfInputFrames;
		const auto frameRelativeIndex = outputFrameIndex % numberOfInputFrames;
		return frameGroupIndex % 2 != 0 ? numberOfInputFrames - frameRelativeIndex - 1
			: frameRelativeIndex;

	}

	void Pipeline::computeView(int inputFrame, int virtualFrame, int virtualView, std::shared_ptr<View> inputImage, std::future<std::shared_ptr<View>>& nextImageFuture, std::future<void>& saveColorFuture)
	{
		auto params_virtual = getConfig().params_virtual[virtualView];

		if (!getConfig().pose_trace.empty())
		{
			auto pose = getConfig().pose_trace[inputFrame];

			params_virtual.setPosition(params_virtual.getPosition() + pose.position);
			params_virtual.setRotation(pose.rotation);

			std::cout << "Pose: " << params_virtual.getPosition() << ' ' << params_virtual.getRotation() << std::endl;
		}

#if WITH_OPENGL
		auto intermediateSize = cv::Size
		(
			int(detail::g_rescale*params_virtual.getSize().width),
			int(detail::g_rescale*params_virtual.getSize().height)
		);

		if (g_with_opengl)
		{
			auto FBO = opengl::RFBO::getInstance();
			FBO->init(intermediateSize);
		}
#endif

		std::unique_ptr<BlendedView> blender = createBlender(virtualView);
		std::unique_ptr<SynthesizedView> synthesizer;
		std::unique_ptr<SpaceTransformer> spaceTransformer = createSpaceTransformer(virtualView);
		spaceTransformer->set_targetPosition(&params_virtual);

		for (auto inputView = 0u; inputView != getConfig().InputCameraNames.size(); ++inputView)
		{
			std::cout << getConfig().InputCameraNames[inputView] << " => " << getConfig().VirtualCameraNames[virtualView] << std::endl;

			auto const& params_real = getConfig().params_real[inputView];

			spaceTransformer->set_inputPosition(&params_real);

			synthesizer = createSynthesizer(inputView, virtualView);
			synthesizer->setSpaceTransformer(spaceTransformer.get());

			int frame_to_load = getExtendedIndex(inputFrame, getConfig().number_of_frames);

			if (inputView == 0 && virtualFrame == 0)
			{
				std::cout << "LOADING       : frame " << frame_to_load << " from " << getConfig().InputCameraNames[inputView] << std::endl;
				inputImage = loadInputView(frame_to_load, inputView, params_real);
			}
			else if (nextImageFuture.valid())
			{
				inputImage = nextImageFuture.get();
			}

			if (inputView < getConfig().InputCameraNames.size() - 1)
			{
				std::cout << "LOADING Async : frame " << frame_to_load << " from " << getConfig().InputCameraNames[inputView + 1] << std::endl;
				nextImageFuture = std::async(std::launch::async,
					[this, frame_to_load, inputView]()
					{
						return loadInputView(frame_to_load, inputView + 1, getConfig().params_real[inputView + 1]);
					}
				);
			}
			else if (inputView == getConfig().InputCameraNames.size() - 1 && virtualFrame < getConfig().number_of_output_frames - 1)
			{
				int next_frame_to_load = getExtendedIndex(inputFrame + 1, getConfig().number_of_frames);
				std::cout << "LOADING Async : frame " << next_frame_to_load << " from " << getConfig().InputCameraNames[0] << std::endl;
				nextImageFuture = std::async(std::launch::async,
					[this, next_frame_to_load, inputView]()
					{
						return loadInputView(next_frame_to_load, 0, getConfig().params_real[0]);
					}
				);
			}

#if WITH_OPENGL
			if (g_with_opengl)
				opengl::rd_start_capture_frame();
#endif

			synthesizer->compute(*inputImage);

			blender->blend(*synthesizer);

#if WITH_OPENGL
			if (g_with_opengl)
				opengl::rd_end_capture_frame();
#endif
			if (inputView == getConfig().InputCameraNames.size() - 1)
			{
				inputImage.reset();
				if (g_with_cuda)
					blender->freeMemory(synthesizer->getOutputSize());
			}
		}

#if WITH_OPENGL
		if (g_with_opengl)
			blender->assignFromGL2CV(intermediateSize);
#endif

		cv::Mat3f color;
		float3* devColor;
		if (!g_with_cuda)
		{
			color = detail::inpaintBasicMode(*blender);

			if (wantColor())
				saveColor(color, virtualFrame, virtualView, params_virtual);
		}
		else
		{
			devColor = detail::inpaintCudaMode(*blender);

			cv::Size size = blender->getOutputSize();

			if (saveColorFuture.valid())
				saveColorFuture.get();

			if (wantColor())
				saveColorFuture = std::async(std::launch::async,
					[this, devColor, virtualFrame, virtualView, params_virtual, size]()
					{
						return saveColor(devColor, virtualFrame, virtualView, params_virtual, size);
					});
		}

		// If CUDA: then exit func : next files useless.
		if (g_with_cuda)
			return;

		// Compute mask (activated by OutputMasks or MaskedOutputFiles or MaskedDepthOutputFiles)
		cv::Mat1b mask;
		if (wantMask() || wantMaskedColor() || wantMaskedDepth())
		{
			mask = blender->get_validity_mask(getConfig().validity_threshold);
			resize(mask, mask, params_virtual.getSize(), cv::INTER_NEAREST);
		}

		// Write mask (activated by OutputMasks)
		if (wantMask())
			saveMask(mask, virtualFrame, virtualView, params_virtual);

		// Write masked output (activated by MaskedOutputFiles)
		if (wantMaskedColor())
		{
			color.setTo(cv::Vec3f::all(0.5f), mask);
			saveMaskedColor(color, virtualFrame, virtualView, params_virtual);
		}

		// Write depth maps (activated by DepthOutputFiles)
		if (wantDepth())
		{
			auto depth = blender->get_depth();
			resize(depth, depth, params_virtual.getSize());
			saveDepth(depth, virtualFrame, virtualView, params_virtual);
		}

		// Write masked depth maps (activated by MaskedDepthOutputFiles)
		if (wantMaskedDepth())
		{
			auto depth = blender->get_depth();
			resize(depth, depth, params_virtual.getSize());
			saveMaskedDepth(depth, mask, virtualFrame, virtualView, params_virtual);
		}

#if WITH_OPENGL
		if (g_with_opengl)
		{
			auto FBO = opengl::RFBO::getInstance();
			FBO->free();
		}
#endif
	}

	std::unique_ptr<BlendedView> Pipeline::createBlender(int)
	{
		if (getConfig().blending_method == BlendingMethod::simple)
		{
			auto temp = new BlendedViewSimple(getConfig().blending_factor);
			return std::unique_ptr<BlendedView>(temp);
		}

		if (getConfig().blending_method == BlendingMethod::multispectral)
			return std::unique_ptr<BlendedView>(new BlendedViewMultiSpec(getConfig().blending_low_freq_factor, getConfig().blending_high_freq_factor));

		std::ostringstream what;
		what << "Unknown view blending method \"" << getConfig().blending_method << "\"";
		throw std::runtime_error(what.str());
	}

	std::unique_ptr<SynthesizedView> Pipeline::createSynthesizer(int, int)
	{
		if (getConfig().vs_method == ViewSynthesisMethod::triangles)
			return std::unique_ptr<SynthesizedView>(new SynthetisedViewTriangle);

		std::ostringstream what;
		what << "Unknown view synthesis method \"" << getConfig().vs_method << "\"";
		throw std::runtime_error(what.str());
	}

	std::unique_ptr<SpaceTransformer> Pipeline::createSpaceTransformer(int)
	{
#if WITH_OPENGL
		if (g_with_opengl)
			return std::unique_ptr<SpaceTransformer>(new OpenGLTransformer);
#endif
		return std::unique_ptr<SpaceTransformer>(new GenericTransformer);
	}
}