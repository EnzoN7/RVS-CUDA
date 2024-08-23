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

#include "inpainting.hpp"
#include "Config.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

namespace rvs
{
	namespace detail
	{
		cv::Mat compute_mask(std::vector<cv::Mat> depths) {
			cv::Mat mask = cv::Mat(depths[0].size(), CV_8U, cv::Scalar::all(255));
			for (int i = 0; i < static_cast<int>(depths.size()); i++) {
				cv::Mat tmp = depths[i] == 0.f;
				bitwise_and(tmp, mask, mask);
			}
			return mask;
		}
		cv::Mat raffine_mask(cv::Mat mask) {
			cv::Mat mask2;
			erode(mask, mask2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3 * (int)g_rescale, 3 * (int)g_rescale)), cv::Point(-1, -1), 1);
			dilate(mask2, mask2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3 * (int)g_rescale, 3 * (int)g_rescale)), cv::Point(-1, -1), 1);
			return mask2;
		}


		void compute_interpolation(cv::Mat & map, int dstmax) { //map : 4 channel distXneg, distXpos, distYneg, distYpos = distance to the nearest known pixel
			for (int x = 0; x < map.cols; x++)
				for (int y = 0; y < map.rows; y++)
				{
					cv::Vec4i pos = map.at<cv::Vec4i>(y, x);
					if (pos[0] == 0)
					{
						for (int i = x - 1; i > -1 && x - i < dstmax; --i) {
							if (map.at<cv::Vec4i>(y, i)[0] == 0)
								break;
							map.at<cv::Vec4i>(y, i)[0] = x - i;
						}
						for (int i = x + 1; i < map.cols && i - x < dstmax; ++i) {
							if (map.at<cv::Vec4i>(y, i)[1] == 0)
								break;
							map.at<cv::Vec4i>(y, i)[1] = i - x;
						}
						for (int j = y - 1; j > -1 && y - j < dstmax; --j) {
							if (map.at<cv::Vec4i>(j, x)[2] == 0)
								break;
							map.at<cv::Vec4i>(j, x)[2] = y - j;
						}
						for (int j = y + 1; j < map.rows && j - y < dstmax; ++j) {
							if (map.at<cv::Vec4i>(j, x)[3] == 0)
								break;
							map.at<cv::Vec4i>(j, x)[3] = j - y;
						}
					}
				}
		}
		void compute_nearest(cv::Mat & map) { // map : 3 channels : x of the nearest,y of the nearest,dist
			bool change = true;
			bool odd = true;
			while (change) {
				odd = !odd;
				change = false;
				int A = 0; int B = 1;
				int C = 0; int D = 1;
				if (odd) {
					A = map.cols - 1; B = -1;
					C = map.rows - 1; D = -1;
				}
				for (int x = A; x < map.cols && x > -1; x += B)
					for (int y = C; y < map.rows && y > -1; y += D)
					{
						cv::Vec3i pos = map.at<cv::Vec3i>(y, x);
						if (pos[2] > 0)
						{
							for (int i = std::max(x - 1, 0); i < std::min(x + 2, map.cols); ++i)
								for (int j = std::max(y - 1, 0); j < std::min(y + 2, map.rows); ++j)
								{
									if (abs(x - i) + abs(y - j) == 1) {
										cv::Vec3i path = map.at<cv::Vec3i>(j, i);
										if (path[2] + 1 < map.at<cv::Vec3i>(y, x)[2]) {
											map.at<cv::Vec3i>(y, x) = cv::Vec3i(path[0], path[1], path[2] + 1);
											change = true;
										}
									}
								}
						}
					}
			}
		}
		void inpaint_color(const cv::Mat& src, const cv::Mat & mask, cv::Mat& dst) {
			bool compute_by_interpolation = false;
			bool compute_by_nearest = true;
			if (compute_by_interpolation) {
				//inpainting from nearest in each direction
				cv::Mat map = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32SC4);
				for (int x = 0; x < map.cols; x++)
					for (int y = 0; y < map.rows; y++)
					{
						map.at<cv::Vec4i>(y, x) = cv::Vec4i(0, 0, 0, 0);
						if (mask.at<bool>(y, x))
							map.at<cv::Vec4i>(y, x) = cv::Vec4i(map.cols + map.rows, map.cols + map.rows, map.cols + map.rows, map.cols + map.rows);
					}
				compute_interpolation(map, (int)(1000.0f*g_rescale));
				for (int y = 0; y < src.rows; ++y)
					for (int x = 0; x < src.cols; ++x) {
						if (mask.at<bool>(y, x)) {
							cv::Vec4i pix = map.at <cv::Vec4i>(y, x);
							if (pix[0] == 0)
								continue;
							cv::Vec3f d = 0;
							float s = 0;
							if (x + pix[0] < map.cols) {
								d += 1.0f / (pix[0] * pix[0]) * dst.at<cv::Vec3f>(y, x + pix[0]);
								s += 1.0f / (pix[0] * pix[0]);
							}
							if (x - pix[1] > -1) {
								d += 1.0f / pix[1] / pix[1] * dst.at<cv::Vec3f>(y, x - pix[1]);
								s += 1.0f / pix[1] / pix[1];
							}
							if (y + pix[2] < map.rows) {
								d += 1.0f / pix[2] / pix[2] * dst.at<cv::Vec3f>(y + pix[2], x);
								s += 1.0f / pix[2] / pix[2];
							}
							if (y - pix[3] > -1) {
								d += 1.0f / pix[3] / pix[3] * dst.at<cv::Vec3f>(y - pix[3], x);
								s += 1.0f / pix[3] / pix[3];
							}
							dst.at <cv::Vec3f>(y, x) = d / s;
						}
					}
			}
			else if (compute_by_nearest) {
					//inpainting by nearest
					cv::Mat map = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32SC3);
					for (int x = 0; x < map.cols; x++)
						for (int y = 0; y < map.rows; y++)
						{
							map.at<cv::Vec3i>(y, x) = cv::Vec3i(x, y, 0);
							if (mask.at<bool>(y, x))
								map.at<cv::Vec3i>(y, x)[2] = map.cols + map.rows;
						}
					compute_nearest(map);
					cv::Mat xyz[3];
					cv::split(map, xyz);
					for (int y = 0; y < src.rows; ++y)
						for (int x = 0; x < src.cols; ++x) {
							if (mask.at<bool>(y, x)) {
								cv::Vec3i pix = map.at <cv::Vec3i>(y, x);
								dst.at<cv::Vec3f>(y, x) = dst.at <cv::Vec3f>(pix[1], pix[0]);
							}
						}
			}
			else {
				//inpainting en lignes
				cv::Vec3f nearestcol(0, 0, 0);
				for (int y = 0; y < src.rows; ++y)
					for (int x = 0; x < src.cols; ++x) {
						if (mask.at<bool>(y, x))
							dst.at<cv::Vec3f>(y, x) = nearestcol;
						else
							nearestcol = dst.at<cv::Vec3f>(y, x);
					}
			}
		}

		cv::Mat inpaintBasicMode(View blender)
		{
			const cv::Mat img = blender.get_color();
			const cv::Mat mask = blender.get_inpaint_mask();

			cv::Mat3f inpainted = img;

			inpaint_color(img, mask, inpainted);

			return inpainted;
		}

		float3* inpaintCudaMode(View blender)
		{

			float3* color = blender.getDevColor();
			cv::Size size = blender.getOutputSize();

			try { inpaintImg(color, size); }
			catch (const std::runtime_error& e) { std::cerr << "CUDA Error (inpaint) : " << e.what() << std::endl; }

			return color;
		}

		//renvoie un inpainting complet. renvoie une image dans le type demande
		cv::Mat inpaint_all(const cv::Mat& img, const cv::Mat& prev, int return_type, int cvt_type, int col_cvt_type, int col_cvtback_type, cv::Vec3f empty_color) {
			cv::Mat img_8;
			cv::Mat inpaint_mask = prev == 0.f;
			cv::Mat mask = raffine_mask(inpaint_mask);
			if (col_cvt_type > -1)
				cvtColor(img, img_8, col_cvt_type);
			else
				img.copyTo(img_8);
			if (return_type == CV_32F || return_type == CV_32FC3 || return_type == CV_32FC4)
				img_8 *= 255.0;
			img_8.convertTo(img_8, cvt_type);
			cv::Mat inpainted = cv::Mat::zeros(img_8.size(), cvt_type);
			inpaint(img_8, inpaint_mask, inpainted, 1, cv::INPAINT_NS);
			if (col_cvtback_type > -1)
				cvtColor(inpainted, inpainted, col_cvtback_type);
			inpainted.convertTo(inpainted, return_type);
			if (return_type == CV_32F || return_type == CV_32FC3 || return_type == CV_32FC4)
				inpainted /= 255.0;
			if (return_type == CV_32FC4)
				inpainted.setTo(empty_color, mask);
			if (return_type == CV_32F)
				inpainted.setTo(0, mask);
			return inpainted;
		}
	}
}
