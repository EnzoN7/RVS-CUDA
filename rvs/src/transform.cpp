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

#include "transform.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <algorithm>

namespace rvs
{
	namespace detail
	{
		extern float g_rescale; // Config.hpp

		namespace
		{
			float valid_tri(cv::Vec2f A, cv::Vec2f B, cv::Vec2f C) {
				double dab = cv::norm(A, B);
				double dac = cv::norm(A, C);
				double dbc = cv::norm(B, C);

				float stretch = static_cast<float>(std::max(dbc, std::max(dab, dac)));
				stretch /= g_rescale;

				float quality = 10000.f - 1000.f * stretch;
				quality = std::max(1.f, quality); // always > 0
				quality = std::min(10000.f, quality);

				return quality;
			}

			void colorize_triangle(const cv::Mat & img, const cv::Mat & depth, const cv::Mat& depth_prologation_mask, const cv::Mat & new_pos, cv::Mat& res, cv::Mat& new_depth, cv::Mat& new_depth_prologation_mask, cv::Mat& triangle_shape, cv::Point a, cv::Point b, cv::Point c) {
				cv::Vec2f A = new_pos.at<cv::Vec2f>(a);
				cv::Vec2f B = new_pos.at<cv::Vec2f>(b);
				cv::Vec2f C = new_pos.at<cv::Vec2f>(c);

				float den = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]);
				if (den <= 0.f)
					return;

				float triangle_validity = valid_tri(A, B, C);
				if (triangle_validity == 0.f)
					return;

				auto Xmin = std::max(0, static_cast<int>(std::floor(std::min(std::min(A[0], B[0]), C[0]))));
				auto Ymin = std::max(0, static_cast<int>(std::floor(std::min(std::min(A[1], B[1]), C[1]))));
				auto Xmax = std::min(res.cols - 1, static_cast<int>(std::ceil(std::max(std::max(A[0], B[0]), C[0]))));
				auto Ymax = std::min(res.rows - 1, static_cast<int>(std::ceil(std::max(std::max(A[1], B[1]), C[1]))));	

				if (Ymin > Ymax || Xmin > Xmax)
					return;

				cv::Vec3f colA = img.at<cv::Vec3f>(a);
				cv::Vec3f colB = img.at<cv::Vec3f>(b);
				cv::Vec3f colC = img.at<cv::Vec3f>(c);
				float dA = depth.at<float>(a);
				float dB = depth.at<float>(b);
				float dC = depth.at<float>(c);

				for (int y = Ymin; y <= Ymax; ++y)
					for (int x = Xmin; x <= Xmax; ++x)
					{
						float const offset = 0.5f;
						float lambda_1 = ((B[1] - C[1]) * ((float)x + offset - C[0]) + (C[0] - B[0]) * ((float)y + offset - C[1])) / den;
						float lambda_2 = ((C[1] - A[1]) * ((float)x + offset - C[0]) + (A[0] - C[0]) * ((float)y + offset - C[1])) / den;
						float lambda_3 = 1.f - lambda_1 - lambda_2;

						float const eps = 1e-6f;
						if (lambda_1 >= -eps && lambda_2 >= -eps && lambda_3 >= -eps)
						{
							cv::Vec3f col = colA * lambda_1 + colB * lambda_2 + colC * lambda_3;
							float d = dA * lambda_1 + dB * lambda_2 + dC * lambda_3;

							auto& new_d = new_depth.at<float>(y, x);
							auto& shape = triangle_shape.at<float>(y, x);
							auto ratio = d / new_d;
							auto is_valid = ratio * ratio * ratio * shape < triangle_validity;
							auto& new_prol = new_depth_prologation_mask.at<bool>(y, x);
							auto prol = depth_prologation_mask.at<bool>(a)
								|| depth_prologation_mask.at<bool>(b)
								|| depth_prologation_mask.at<bool>(c);

							//if the pixel comes from original depth map and is in foreground
							if ((is_valid || new_prol) && !prol)
							{
								new_d = d;
								new_prol = false;
								shape = triangle_validity;
								res.at<cv::Vec3f>(y, x) = col;
							}
							//if the pixel comes from inpainted depth map and is in foreground and there is no pixel from the original depth map
							else if (is_valid && new_prol && prol)
							{
								new_d = d;
								shape = triangle_validity;
								res.at<cv::Vec3f>(y, x) = col;
							}
						}
					}
			}
		} // namespace

		cv::Mat3f transform_trianglesMethod(cv::Mat3f input_color, cv::Mat1f input_depth, cv::Mat2f input_positions, cv::Size output_size, cv::Mat1f& depth, cv::Mat1f& triangle_shape, bool horizontalWrap)
		{
			auto input_size = input_color.size();
			int w = input_size.width;
			cv::Mat1b input_depth_mask = input_depth > 0.f;
			cv::Mat3f color = cv::Mat3f::zeros(output_size);
			depth = cv::Mat1f(output_size, std::numeric_limits<float>::infinity());
			triangle_shape = cv::Mat1f::zeros(output_size);
			cv::Mat1b new_depth_prologation_mask = cv::Mat1b::ones(output_size);

			// Compute triangulation
			for (int i = 0; i < input_size.height - 1; ++i) {
				for (int j = 0; j < input_size.width - 1; ++j) {
					if (input_depth(i, j + 1) > 0.f && input_depth(i + 1, j) > 0.f && /*why?*/ input_positions(i, j + 1)[0] > 0.f && /*why?*/ input_positions(i + 1, j)[0] > 0.f) {
						if (input_depth(i, j) > 0.f && /*why?*/ input_positions(i, j)[0] > 0.f)
							colorize_triangle(input_color, input_depth, input_depth_mask, input_positions, color, depth, new_depth_prologation_mask, triangle_shape,
								cv::Point(j, i), cv::Point(j + 1, i), cv::Point(j, i + 1));
						if (input_depth(i + 1, j + 1) > 0.f && input_positions(i + 1, j + 1)[0] > 0.f)
							colorize_triangle(input_color, input_depth, input_depth_mask, input_positions, color, depth, new_depth_prologation_mask, triangle_shape,
								cv::Point(j + 1, i + 1), cv::Point(j, i + 1), cv::Point(j + 1, i));

						// stitch left and right borders with triangles (e.g. for equirectangular projection)
						if (horizontalWrap && j == 0) {
							colorize_triangle(input_color, input_depth, input_depth_mask, input_positions, color, depth, new_depth_prologation_mask, triangle_shape,
								cv::Point(w - 1, i), cv::Point(0, i), cv::Point(w - 1, i + 1));
							colorize_triangle(input_color, input_depth, input_depth_mask, input_positions, color, depth, new_depth_prologation_mask, triangle_shape,
								cv::Point(0, i + 1), cv::Point(w - 1, i + 1), cv::Point(0, i));
						}
					}
				}
			}

			return color;
		}
	}
}
