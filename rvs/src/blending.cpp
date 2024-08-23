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

#include "blending.hpp"
#include "IntegralImage2D.h"
#include "Config.hpp"

#include <iostream>

#include <opencv2/imgproc.hpp>

namespace rvs
{
	namespace detail
	{
		/**
		 * Simple blur
		 * */
		void calcBlurring(cv::Mat img, cv::Mat &blr, cv::Mat msk, int rad)
		{
			int chn = img.channels();
			if (chn == 1)
			{
				blr = cv::Mat(img.size().height, img.size().width, CV_32FC1);

				int srd_ele = 1;
				int srd_row = srd_ele * img.size().width;

				IntegralImage2D<float, 1> iim_dx;
				iim_dx.setInput((float*)img.data, msk.data, img.size().width, img.size().height, srd_ele, srd_row);

				int rad_2 = rad;
				int rad_1 = int(2 * rad_2 + 1);
#pragma omp parallel default(shared)
				{
#pragma omp for schedule(dynamic, 10) nowait
					for (int i = 0; i < img.size().height; i++)
					{
						for (int j = 0; j < img.size().width; j++)
						{
							if (msk.at<uchar>(i, j) == 0)
							{
								blr.at<float>(i, j) = 0;
								continue;
							}


							int start_x = j - rad_2;
							int start_y = i - rad_2;
							int end_y = start_y + rad_1;
							int end_x = start_x + rad_1;

							start_x = std::max(0, start_x);
							start_y = std::max(0, start_y);
							end_x = std::min(end_x, img.size().width);
							end_y = std::min(end_y, img.size().height);

							unsigned cnt = iim_dx.getFiniteElementsCountSE(start_x, start_y, end_x, end_y);
							unsigned sum = (int)iim_dx.getFirstOrderSumSE(start_x, start_y, end_x, end_y);

							if (cnt > 0)
								blr.at<float>(i, j) = float(float(sum) / float(cnt));
						}
					}
				}
			}
			else if (chn == 3)
			{
				blr = cv::Mat(img.size().height, img.size().width, CV_32FC3);

				int srd_ele = sizeof(cv::Vec3f) / sizeof(float);
				int srd_row = srd_ele * img.size().width;

				IntegralImage2D<float, 3> iim_dx;
				iim_dx.setInput((float*)img.data, msk.data, img.size().width, img.size().height, srd_ele, srd_row);

				int rad_2 = rad;
				int rad_1 = int(2 * rad_2 + 1);
#pragma omp parallel default(shared)
				{
#pragma omp for schedule(dynamic, 10) nowait
					for (int i = 0; i < img.size().height; i++)
					{
						for (int j = 0; j < img.size().width; j++)
						{
							if (msk.at<uchar>(i, j) == 0)
							{
								blr.at<cv::Vec3f>(i, j)[0] = 0;
								blr.at<cv::Vec3f>(i, j)[1] = 0;
								blr.at<cv::Vec3f>(i, j)[2] = 0;
								continue;
							}

							int start_x = j - rad_2;
							int start_y = i - rad_2;
							int end_y = start_y + rad_1;
							int end_x = start_x + rad_1;

							start_x = std::max(0, start_x);
							start_y = std::max(0, start_y);
							end_x = std::min(end_x, img.size().width);
							end_y = std::min(end_y, img.size().height);

							unsigned count = iim_dx.getFiniteElementsCountSE(start_x, start_y, end_x, end_y);
							cv::Vec3f sum = iim_dx.getFirstOrderSumSE(start_x, start_y, end_x, end_y);

							if (count > 0) {
								blr.at<cv::Vec3f>(i, j)[0] = float(float(sum[0]) / float(count));
								blr.at<cv::Vec3f>(i, j)[1] = float(float(sum[1]) / float(count));
								blr.at<cv::Vec3f>(i, j)[2] = float(float(sum[2]) / float(count));
							}
						}
					}
				}
			}
		}
		/**
		 * gaussian blur
		 * */
		void calcBlurringGaussian(cv::Mat img, cv::Mat &blr, cv::Mat msk, int rad, float ect = 0.0f)
		{
			float sig = (ect == 0.0f) ? 0.3f * float(rad) + 0.5f : ect;
			std::vector<float> fct_vec(rad + 1);
			for (int rad_idx = rad; rad_idx >= 0; rad_idx--)
			{
				fct_vec[rad_idx] = std::exp(-float(rad_idx * rad_idx) / (2.0f * sig * sig));
				if (rad_idx < rad)
					fct_vec[rad_idx] -= fct_vec[rad_idx + 1];
			}

			int chn = img.channels();
			if (chn == 1)
			{
				blr = cv::Mat(img.size().height, img.size().width, CV_8UC1);

				int srd_ele = 1;
				int srd_row = srd_ele * img.size().width;

				IntegralImage2D<unsigned char, 1> iim_dx;
				iim_dx.setInput(img.data, msk.data, img.size().width, img.size().height, srd_ele, srd_row);

#pragma omp parallel default(shared)
				{
#pragma omp for schedule(dynamic, 10) nowait
					for (int i = 0; i < img.size().height; i++)
					{
						for (int j = 0; j < img.size().width; j++)
						{
							if (msk.at<uchar>(i, j) == 0)
							{
								blr.at<uchar>(i, j) = 0;
								continue;
							}

							float cnt_tot = 0.0f;
							float sum_tot = 0.0f;
							for (int rad_idx = rad; rad_idx >= 0; rad_idx--)
							{
								int start_x = j - rad_idx;
								int start_y = i - rad_idx;
								int end_y = start_y + 2 * rad_idx + 1;
								int end_x = start_x + 2 * rad_idx + 1;

								start_x = std::max(0, start_x);
								start_y = std::max(0, start_y);
								end_x = std::min(end_x, img.size().width);
								end_y = std::min(end_y, img.size().height);

								unsigned cnt = iim_dx.getFiniteElementsCountSE(start_x, start_y, end_x, end_y);
								unsigned sum = iim_dx.getFirstOrderSumSE(start_x, start_y, end_x, end_y);

								cnt_tot += fct_vec[rad_idx] * float(cnt);
								sum_tot += fct_vec[rad_idx] * float(sum);
							}
							if (cnt_tot > 0)
								blr.at<uchar>(i, j) = uchar(float(sum_tot) / float(cnt_tot));
						}
					}
				}
			}
			else if (chn == 3)
			{
				blr = cv::Mat(img.size().height, img.size().width, CV_8UC3);

				int srd_ele = sizeof(cv::Vec3b) / sizeof(uchar);
				int srd_row = srd_ele * img.size().width;

				IntegralImage2D<unsigned char, 3> iim_dx;
				iim_dx.setInput(img.data, msk.data, img.size().width, img.size().height, srd_ele, srd_row);

#pragma omp parallel default(shared)
				{
#pragma omp for schedule(dynamic, 10) nowait
					for (int i = 0; i < img.size().height; i++)
					{
						for (int j = 0; j < img.size().width; j++)
						{
							if (msk.at<uchar>(i, j) == 0)
							{
								blr.at<cv::Vec3b>(i, j)[0] = 0;
								blr.at<cv::Vec3b>(i, j)[1] = 0;
								blr.at<cv::Vec3b>(i, j)[2] = 0;
								continue;
							}

							float cnt_tot = 0.0f;
							cv::Vec3f sum_tot = cv::Vec3f(0.0f, 0.0f, 0.0f);
							for (int rad_idx = rad; rad_idx >= 0; rad_idx--)
							{
								int start_x = j - rad_idx;
								int start_y = i - rad_idx;
								int end_y = start_y + 2 * rad_idx + 1;
								int end_x = start_x + 2 * rad_idx + 1;

								start_x = std::max(0, start_x);
								start_y = std::max(0, start_y);
								end_x = std::min(end_x, img.size().width);
								end_y = std::min(end_y, img.size().height);

								unsigned cnt = iim_dx.getFiniteElementsCountSE(start_x, start_y, end_x, end_y);
								cv::Vec3i sum = iim_dx.getFirstOrderSumSE(start_x, start_y, end_x, end_y);

								cnt_tot += fct_vec[rad_idx] * float(cnt);
								sum_tot[0] += fct_vec[rad_idx] * float(sum[0]);
								sum_tot[1] += fct_vec[rad_idx] * float(sum[1]);
								sum_tot[2] += fct_vec[rad_idx] * float(sum[2]);
							}
							if (cnt_tot > 0)
							{
								blr.at<cv::Vec3b>(i, j)[0] = uchar(sum_tot[0] / cnt_tot);
								blr.at<cv::Vec3b>(i, j)[1] = uchar(sum_tot[1] / cnt_tot);
								blr.at<cv::Vec3b>(i, j)[2] = uchar(sum_tot[2] / cnt_tot);
							}
						}
					}
				}
			}
		}
		void calcBlurringBilateral(cv::Mat img, cv::Mat &blr, cv::Mat /*msk*/, int bil_rad = 10, int bil_itr = 10, int bil_sig_col = 10, int bil_sig_spc = 150)
		{
			blr = img.clone();
			for (int itr = 0; itr < bil_itr; itr++)
			{
				cv::Mat out;
				cv::bilateralFilter(blr, out, bil_rad, bil_sig_col, bil_sig_spc);
				blr = out;
			}
		}

		cv::Mat blend_img_by_max(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& qualities, const std::vector<cv::Mat>& depth_prolongations, cv::Vec3f empty_color, cv::Mat& quality, cv::Mat& depth_prolongation_mask, cv::Mat& inpaint_mask) {
			cv::Mat3f color = cv::Mat3f(imgs.front().size(), empty_color);
			quality = cv::Mat1f::zeros(imgs[0].size());
			depth_prolongation_mask = cv::Mat1b::zeros(imgs[0].size());
			inpaint_mask = cv::Mat1b::zeros(imgs[0].size());

			for (std::size_t i = 0; i != imgs.size(); ++i) {
				cv::Mat1b mask = qualities[i] > quality;
				imgs[i].copyTo(color, mask);
				qualities[i].copyTo(quality, mask);
				depth_prolongations[i].copyTo(depth_prolongation_mask, mask);
				inpaint_mask.setTo(false, mask);
			}

			return color;
		}

		cv::Mat blend_img(const std::vector<cv::Mat> colors, const std::vector<cv::Mat> qualities, const std::vector<cv::Mat> depthProlongations, cv::Vec3f emptyColor, cv::Mat& outputQuality, cv::Mat& outputDepthProlongationMask, cv::Mat& outputInpaintMask, float blendingExponent) {
			if (blendingExponent < 0)
				return blend_img_by_max(colors, qualities, depthProlongations, emptyColor, outputQuality, outputDepthProlongationMask, outputInpaintMask);

			cv::Mat blendedImage = cv::Mat::zeros(colors[0].size(), CV_32FC3);
			cv::Mat qualityResult = cv::Mat::zeros(colors[0].size(), CV_32F);
			cv::Mat depthProlongationMaskResult = cv::Mat::zeros(colors[0].size(), outputDepthProlongationMask.type());
			cv::Mat inpaintMaskResult = (qualityResult < -1.0);

			for (int y = 0; y < blendedImage.rows; ++y) {
				for (int x = 0; x < blendedImage.cols; ++x) {
					float sumWeights = 0.0;
					cv::Vec3f inpaintedColor = emptyColor;
					cv::Vec3f inpaintedDepthColor(0, 0, 0);
					cv::Vec3f finalColor(0, 0, 0);
					float inpaintedDepthSumWeights = 0.0;

					for (int i = 0; i < static_cast<int>(colors.size()); ++i) {
						float quality = qualities[i].at<float>(y, x);
						bool isValidDepth = qualities[i].at<float>(y, x) > 0 && !depthProlongations[i].at<bool>(y, x);

						if (quality > 0 && isValidDepth) {
							quality = powf(quality, blendingExponent);
							sumWeights += quality;
							finalColor += quality * colors[i].at<cv::Vec3f>(y, x);
						}
						else if (quality > 0 && qualities[i].at<float>(y, x) > 0) {
							quality = powf(quality, blendingExponent);
							inpaintedDepthSumWeights += quality;
							inpaintedDepthColor += quality * colors[i].at<cv::Vec3f>(y, x);
						}
					}

					if (sumWeights == 0) {
						if (inpaintedDepthSumWeights == 0.0) {
							blendedImage.at<cv::Vec3f>(y, x) = inpaintedColor;
							inpaintMaskResult.at<bool>(y, x) = true;
							depthProlongationMaskResult.at<bool>(y, x) = true;
							qualityResult.at<float>(y, x) = 0.0;
						}
						else {
							blendedImage.at<cv::Vec3f>(y, x) = inpaintedDepthColor / inpaintedDepthSumWeights;
							depthProlongationMaskResult.at<bool>(y, x) = true;
							qualityResult.at<float>(y, x) = powf(inpaintedDepthSumWeights, 1.0f / blendingExponent);
						}
					}
					else {
						blendedImage.at<cv::Vec3f>(y, x) = finalColor / sumWeights;
						depthProlongationMaskResult.at<bool>(y, x) = false;
						qualityResult.at<float>(y, x) = powf(sumWeights, 1.0f / blendingExponent);
					}
				}
			}

			inpaintMaskResult.copyTo(outputInpaintMask);
			outputDepthProlongationMask = depthProlongationMaskResult;
			outputQuality = qualityResult;

			return blendedImage;
		}

		void split_frequencies(const cv::Mat & img, cv::Mat & low_freq, cv::Mat & high_freq, const cv::Mat& mask)
		{
			int kernel_size = ((int)MAX(img.rows, img.cols) / 20);
			//RGB: blur all three channels
			if (g_color_space == ColorSpace::RGB) {
				cv::Mat img_rgb_blurry;

				calcBlurring(img, low_freq, mask, kernel_size);

				high_freq = img - low_freq;
			}
			//YCrCb: blur only Y channel
			else if (g_color_space == ColorSpace::YUV) {
				CV_DbgAssert(img.channels() == 3);
				cv::Mat chans[3];
				cv::split(img, chans);

				calcBlurring(chans[0], chans[0], mask, kernel_size);

				cv::merge(chans, 3, low_freq);

				high_freq = img - low_freq;
			}
		}
	}
}

