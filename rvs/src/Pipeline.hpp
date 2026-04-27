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

#ifndef _PIPELINE_HPP_
#define _PIPELINE_HPP_

#include "View.hpp"
#include "Config.hpp"
#include "BlendedView.hpp"
#include "SynthesizedView.hpp"
#include "PoseTraces.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <future>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace rvs
{
	struct DuoStream_t
	{
		cudaStream_t str1;
		cudaStream_t str2;
	};

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	class Pipeline
	{
	  public:
		Pipeline() = default;
		virtual ~Pipeline() = default;

		void execute();

	  protected:
		virtual Config const& getConfig() const = 0;

		virtual void loadInputView(std::shared_ptr<InputView<channel_t, color_t>>& inputImage,
								   cudaStream_t& streamCol, cudaStream_t& streamDep) = 0;

		virtual std::unique_ptr<BlendedView<channel_t, color_t>>
		createBlender(int numInputViews,
					  cv::Size virtualSize, cv::Size outputY_size,
					  cv::Size outputUV_size, int outputType,
					  unsigned outputMaxVal, Config config);

		virtual std::unique_ptr<SpaceTransformer> createSpaceTransformer();

		virtual std::unique_ptr<SynthesizedView<position_t, color_t, channel_t>>
		createSynthesizer(cv::Size realSize, cv::Size virtualSize, SpaceTransformer const* object);

		virtual std::shared_ptr<InputView<channel_t, color_t>>
		createInputView(Config config, int view, int frame,
						cv::Size initialSize, cv::Size realSize,
						cv::Size initialUV_size,
						float colorScale, int initialColorsType,
						float depthScale, int initialDepthType,
						size_t inputDepthBytes, size_t yInputBytes,
						size_t uvInputBytes, size_t outputBytes);

	  private:
		void computeView(
			int& inputFrame,
			int& virtualFrame,
			const Config& config,
			int& totalInputFrames,
			int& frame_to_load,
			int& next_frame_to_load,
			int& doubleBufferIndex,
			Pose& pose,
			Parameters& params_virtual,
			const Parameters& initial_param);

		unsigned max_level(int bit_depth);

		int cvdepth_from_bit_depth(int bit_depth);

		int getExtendedIndex(int outputFrameIndex, int numberOfInputFrames);

		void setupExecutionParameters();

		void createStreams();

		void destroyStreams();

		void checkGpuMemoryUsage();

		void computation();

		std::vector<std::unique_ptr<SpaceTransformer>> m_transformers;
		std::vector<std::unique_ptr<SynthesizedView<position_t, color_t, channel_t>>> m_synthesizers;
		std::vector<std::shared_ptr<InputView<channel_t, color_t>>> m_inputImages;
		std::unique_ptr<BlendedView<channel_t, color_t>> m_blender;
		std::vector<std::future<void>> m_synthFutures;
		int m_numInputViews;

		std::vector<cudaStream_t> m_streamsImport;
		std::vector<DuoStream_t> m_streamsSynth;
		std::vector<cudaStream_t> m_streamsExport;

		std::future<void> m_nextImageFuture;
		std::future<void> m_saveColorFuture;
	};
}

#endif
