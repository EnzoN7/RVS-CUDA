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

#include "Pipeline.hpp"

#define MEASURE_EXECUTION_TIME(frames, code_block)                                                                                    \
    do                                                                                                                                \
	{                                                                                                                                 \
        clock_t startTime = clock();                                                                                                  \
        code_block;                                                                                                                   \
        double executeTime = double(clock() - startTime) / CLOCKS_PER_SEC;                                                            \
        std::cout << std::endl                                                                                                        \
                  << "Total time         = " << std::fixed << std::setprecision(4) << executeTime << " sec." << std::endl             \
                  << std::endl                                                                                                        \
                  << "Av. speed          = " << std::fixed << std::setprecision(4) << (frames) / executeTime << " fps." << std::endl  \
                  << "Av. time per frame = " << std::fixed << std::setprecision(4) << executeTime / (frames) << " sec." << std::endl; \
    }                                                                                                                                 \
	while(0)

namespace rvs
{
	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::computation()
	{
		Config config = getConfig();
		Pose pose;
		Parameters& params_virtual = config.params_virtual[0];

		int inputFrame = 0;
		int totalInputFrames = config.number_of_frames;
		int frame_to_load = 0;
		int next_frame_to_load = 0;
		int doubleBufferIndex = 0;

		std::cout << std::endl << "LOADING..." << std::endl;

		MEASURE_EXECUTION_TIME(totalInputFrames,
			{
				for (int virtualFrame = 0; virtualFrame < config.number_of_output_frames; ++virtualFrame)
				{
					computeView(inputFrame, virtualFrame, config, totalInputFrames,
								frame_to_load, next_frame_to_load, doubleBufferIndex, pose, params_virtual);
				}

				if (m_saveColorFuture.valid())
					m_saveColorFuture.get();
			});
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::execute()
	{
		setupExecutionParameters();
		createStreams();
		checkGpuMemoryUsage();

		loadInputView(m_inputImages[0], m_streamsImport[0], m_streamsImport[1]);

		computation();

		destroyStreams();
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::computeView(int& inputFrame,
							   int& virtualFrame,
							   const Config& config,
							   int& totalInputFrames,
							   int& frame_to_load,
							   int& next_frame_to_load,
							   int& doubleBufferIndex,
							   Pose& pose,
							   Parameters& params_virtual)
	{
#ifdef _DEBUG
		std::cout << "FRAME " << virtualFrame << std::endl;
#endif

		inputFrame = config.start_frame + virtualFrame;
		doubleBufferIndex = virtualFrame % 2;

		pose = config.pose_trace[inputFrame];

		params_virtual.setPosition(params_virtual.getPosition() + pose.position);
		params_virtual.setRotation(pose.rotation);

		for (int inputView = 0; inputView < m_numInputViews; ++inputView)
		{
			if (m_nextImageFuture.valid())
				m_nextImageFuture.get();

			if (inputView < m_numInputViews - 1)
			{
				m_nextImageFuture = std::async(std::launch::async,
					[this, inputView, &frame_to_load, &next_frame_to_load]()
					{
						frame_to_load = next_frame_to_load;
						std::shared_ptr<InputView<channel_t, color_t>>& nextView = m_inputImages[inputView + 1];

						nextView->setFrame(frame_to_load);
						loadInputView(nextView, m_streamsImport[0], m_streamsImport[1]);
					}
				);
			}
			else if (virtualFrame < totalInputFrames - 1)
			{
				m_nextImageFuture = std::async(std::launch::async,
					[this, doubleBufferIndex, inputFrame, &next_frame_to_load]()
					{
						next_frame_to_load = (inputFrame + 1) % detail::numFramesSeq;
						std::shared_ptr<InputView<channel_t, color_t>>& firstViewFromNextFrame = m_inputImages[m_numInputViews + doubleBufferIndex];

						firstViewFromNextFrame->setFrame(next_frame_to_load);
						loadInputView(firstViewFromNextFrame, m_streamsImport[0], m_streamsImport[1]);
					}
				);
			}

			m_synthFutures[inputView] = std::async(std::launch::async,
				[this, inputView, virtualFrame, doubleBufferIndex, params_virtual]()
				{
					m_transformers[inputView]->set_targetPosition(&params_virtual);
					m_synthesizers[inputView]->setSpaceTransformer(m_transformers[inputView].get());

					if (inputView == 0 && virtualFrame != 0)
						m_synthesizers[inputView]->compute(m_inputImages[m_numInputViews + (1 - doubleBufferIndex)],
														   m_streamsSynth[0].str1,
														   m_streamsSynth[0].str2);
					else
						m_synthesizers[inputView]->compute(m_inputImages[inputView],
														   m_streamsSynth[inputView].str1,
														   m_streamsSynth[inputView].str2);
				}
			);
		}

		if (m_saveColorFuture.valid())
			m_saveColorFuture.get();

#pragma omp parallel
		{
#pragma omp single nowait
			{
				for (int i = 0; i < m_numInputViews; ++i)
				{
#pragma omp task firstprivate(i)
					{
						color_t* devVirtualColor = nullptr;
						channel_t* devVirtualDepth = nullptr;
						channel_t* devVirtualValidity = nullptr;

						m_synthFutures[i].get();
						m_synthesizers[i]->waitAndGetVirtualComponents(devVirtualColor, devVirtualDepth, devVirtualValidity);
						m_blender->setHostComponents(i, devVirtualColor, devVirtualValidity, devVirtualDepth);
					}
				}
			}
		}

		m_saveColorFuture = std::async(std::launch::async,
			[this]()
			{
				m_blender->blend(m_streamsExport[0], m_streamsExport[1], m_streamsExport[2]);
				m_blender->inpaint(m_streamsExport[0]);
				m_blender->writeColor(m_streamsExport[0], m_streamsExport[1], m_streamsExport[2]);
			}
		);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::setupExecutionParameters()
	{
		const auto& config = getConfig();

		if (config.outfilenames[0].substr(config.outfilenames[0].size() - 4, 4) != ".yuv")
		{
			std::cerr << "Writing frames only in YUV" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (config.pose_trace.empty())
		{
			std::cerr << "Pose trace empty" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (config.VirtualCameraNames.size() != 1)
		{
			std::cerr << "Only 1 virtual camera is acceptable." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		Parameters virtualParams = config.params_virtual[0];
		Parameters realParams = config.params_real[0];

		cv::Size initialSize = realParams.getSize();
		cv::Size initialUV_size = initialSize / 2;
		cv::Size outputY_size = virtualParams.getSize();
		cv::Size outputUV_size = outputY_size / 2;
		cv::Size realSize = initialSize, virtualSize = outputY_size;

		if (detail::g_rescale < 1.0f)
		{
			int alignTo = 32;

			realSize = cv::Size
			(
				((int)std::round(detail::g_rescale * realSize.width) / alignTo) * alignTo,
				((int)std::round(detail::g_rescale * realSize.height) / alignTo) * alignTo
			);

			virtualSize = cv::Size
			(
				((int)std::round(detail::g_rescale * virtualSize.width) / alignTo) * alignTo,
				((int)std::round(detail::g_rescale * virtualSize.height) / alignTo) * alignTo
			);
		}

		m_numInputViews = static_cast<int>(config.InputCameraNames.size());
		if (m_numInputViews < 2)
		{
			std::cerr << "[ERROR] Minimum 2 input views." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		unsigned outputMaxVal = max_level(virtualParams.getColorBitDepth());
		int initialColorsType = CV_MAKETYPE(cvdepth_from_bit_depth(realParams.getColorBitDepth()), 1);
		int initialDepthType = CV_MAKETYPE(cvdepth_from_bit_depth(realParams.getDepthBitDepth()), 1);
		int outputType = CV_MAKETYPE(cvdepth_from_bit_depth(virtualParams.getColorBitDepth()), 1);
		float colorScale = 1.0f / max_level(realParams.getColorBitDepth());
		float depthScale = 1.0f / max_level(realParams.getDepthBitDepth());

		m_transformers.resize(m_numInputViews);
		m_synthesizers.resize(m_numInputViews);
		for (int i = 0; i < m_numInputViews; ++i)
		{
			Config const& cfg = getConfig();

			m_transformers[i] = createSpaceTransformer();
			m_transformers[i]->set_targetPosition(&cfg.params_virtual[0]);
			m_transformers[i]->set_inputPosition(&cfg.params_real[i]);

			m_synthesizers[i] = createSynthesizer(realSize, virtualSize, m_transformers[i].get());
		}

		size_t yReadBytes = initialSize.area() * CV_ELEM_SIZE(initialColorsType);
		size_t uvReadBytes = initialUV_size.area() * CV_ELEM_SIZE(initialColorsType);
		size_t depthReadBytes = initialSize.area() * CV_ELEM_SIZE(initialDepthType);
		size_t depthNormalizedBytes = realSize.area() * sizeof(channel_t);

		m_inputImages.resize(m_numInputViews + 2);
		for (int i = 0; i < m_numInputViews + 2; ++i)
		{
			int view = (i >= m_numInputViews) ? 0 : i;

			if (!(config.texture_names[view].substr(config.texture_names[view].size() - 4, 4) == ".yuv" || config.texture_names[view].substr(config.texture_names[view].size() - 4, 4) == ".mp4"))
			//if (config.texture_names[view].substr(config.texture_names[view].size() - 4, 4) != ".yuv" )
			{
				std::cerr << "Reading frames only in YUV" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			if (config.depth_names[view].substr(config.depth_names[view].size() - 4, 4) != ".yuv")
			{
				std::cerr << "Reading depth only in YUV" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			m_inputImages[i] = createInputView(config, view, 0,
											   initialSize, realSize, initialUV_size,
											   colorScale, initialColorsType,
											   depthScale, initialDepthType,
											   depthReadBytes, yReadBytes,
											   uvReadBytes, depthNormalizedBytes);
		}

		m_blender = createBlender(m_numInputViews, virtualSize, outputY_size,
								  outputUV_size, outputType, outputMaxVal,
								  config);

		m_synthFutures.resize(m_numInputViews);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::createStreams()
	{
		m_streamsSynth.resize(m_numInputViews);
		for (int i = 0; i < m_numInputViews; ++i)
		{
			cudaStreamCreate(&m_streamsSynth[i].str1);
			cudaStreamCreate(&m_streamsSynth[i].str2);
		}

		m_streamsImport.resize(2);
		for (int i = 0; i < m_streamsImport.size(); ++i)
		{
			cudaStreamCreate(&m_streamsImport[i]);
		}

		m_streamsExport.resize(3);
		for (int i = 0; i < m_streamsExport.size(); ++i)
		{
			cudaStreamCreate(&m_streamsExport[i]);
		}
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::destroyStreams()
	{
		for (int i = 0; i < m_numInputViews; ++i)
		{
			cudaStreamDestroy(m_streamsSynth[i].str1);
			cudaStreamDestroy(m_streamsSynth[i].str2);
		}

		for (int i = 0; i < m_streamsImport.size(); ++i)
		{
			cudaStreamDestroy(m_streamsImport[i]);
		}

		for (int i = 0; i < m_streamsExport.size(); ++i)
		{
			cudaStreamDestroy(m_streamsExport[i]);
		}
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	void Pipeline<position_t, color_t, channel_t>::checkGpuMemoryUsage()
	{
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);

		float usagePercent = 100.0f * (1.0f - (float)freeMem / (float)totalMem);
		float threshold1 = 60.0f, threshold2 = 80.0f;

		if (usagePercent < threshold1)
		{
			std::cout << "  * INFO       GPU Memory Usage: " << std::fixed << std::setprecision(2) << usagePercent
				<< "%\n"
				<< std::endl
				<< "----------------------------------------------------------------------------------------" << std::endl;

			std::cout.unsetf(std::ios::fixed);
			std::cout.precision(6);
		}
		else if (usagePercent >= threshold1 && usagePercent < threshold2)
		{
			std::cerr << "  * WARNING    GPU Memory Usage: " << std::fixed << std::setprecision(2) << usagePercent
				<< "% - Consider a GPU with a larger global memory.\n"
				<< std::endl
				<< "----------------------------------------------------------------------------------------" << std::endl;
		}
		else if (usagePercent >= threshold2)
		{
			std::cerr << "  * ERROR      GPU Memory Usage: " << std::fixed << std::setprecision(2) << usagePercent
				<< "% - Consider a GPU with a larger global memory. Exiting to prevent out-of-memory issues!\n"
				<< std::endl
				<< "----------------------------------------------------------------------------------------" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	std::unique_ptr<BlendedView<channel_t, color_t>>
	Pipeline<position_t, color_t, channel_t>::createBlender(int numInputViews,
							cv::Size virtualSize, cv::Size outputY_size,
							cv::Size outputUV_size, int outputType,
							unsigned outputMaxVal, Config config)
	{
		if (getConfig().blending_method == BlendingMethod::simple)
		{
			return std::make_unique<BlendedViewSimple<channel_t, color_t>>(
				numInputViews, virtualSize,
				outputY_size, outputUV_size,
				outputType, outputMaxVal, config);
		}

		std::cerr << "Unknown view blending method \"" << getConfig().blending_method << "\"" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	std::unique_ptr<SynthesizedView<position_t, color_t, channel_t>>
	Pipeline<position_t, color_t, channel_t>::createSynthesizer(cv::Size realSize, cv::Size virtualSize, SpaceTransformer const* object)
	{
		if (getConfig().vs_method == ViewSynthesisMethod::triangles)
			return std::make_unique<rvs::SynthetizedViewTriangle<position_t, color_t, channel_t>>(realSize, virtualSize, object);

		std::cerr << "Unknown view synthesis method \"" << getConfig().vs_method << "\"" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	std::unique_ptr<SpaceTransformer> Pipeline<position_t, color_t, channel_t>::createSpaceTransformer()
	{
		return std::unique_ptr<SpaceTransformer>(new SpaceTransformer);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	std::shared_ptr<InputView<channel_t, color_t>>
	Pipeline<position_t, color_t, channel_t>::createInputView(Config config, int view, int frame,
							  cv::Size initialSize, cv::Size realSize,
							  cv::Size initialUV_size,
						      float colorScale, int initialColorsType,
							  float depthScale, int initialDepthType,
							  size_t inputDepthBytes, size_t yInputBytes,
							  size_t uvInputBytes, size_t outputBytes)
	{
		return std::make_shared<InputView<channel_t, color_t>>(
			config, view, frame,
			initialSize, realSize, initialUV_size,
			colorScale, initialColorsType,
			depthScale, initialDepthType,
			inputDepthBytes, yInputBytes, uvInputBytes, outputBytes);
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	unsigned Pipeline<position_t, color_t, channel_t>::max_level(int bit_depth)
	{
		assert(bit_depth > 0 && bit_depth <= 16);
		return (1u << bit_depth) - 1u;
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	int Pipeline<position_t, color_t, channel_t>::cvdepth_from_bit_depth(int bit_depth)
	{
		if (bit_depth >= 1 && bit_depth <= 8)
			return CV_8U;
		else if (bit_depth >= 9 && bit_depth <= 16)
			return CV_16U;
		else throw std::invalid_argument("invalid raw image bit depth");
	}

	template<typename position_t,
			 typename color_t,
			 typename channel_t>
	int Pipeline<position_t, color_t, channel_t>::getExtendedIndex(int outputFrameIndex, int numberOfInputFrames)
	{

		if (numberOfInputFrames <= 0)
		{
			std::cerr << "Cannot extend frame index with zero input frames" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		const auto frameGroupIndex = outputFrameIndex / numberOfInputFrames;
		const auto frameRelativeIndex = outputFrameIndex % numberOfInputFrames;

		return frameGroupIndex % 2 != 0
			? numberOfInputFrames - frameRelativeIndex - 1
			: frameRelativeIndex;

	}

	template class Pipeline<float2, float3, float>;
	template class Pipeline<double2, double3, double>;
	template class Pipeline<half2, half3, half>;
}