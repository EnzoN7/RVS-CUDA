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
  Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be  Daniele Bonatto, Daniele.Bonatto@ulb.ac.be  Arnaud Schenkel, arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
  Bart Kroon, bart.kroon@philips.com
  Bart Sonneveldt, bart.sonneveldt@philips.com
*/

#ifndef INTEGRALIMAGE2D_H
#define INTEGRALIMAGE2D_H

#include <vector>
#include <opencv2/core.hpp>

#ifndef isNan
  #define isNan(a)(!(a == a))
#endif

namespace rvs
{
	namespace detail
	{
		template <typename DataType> struct TypeTraits
		{
			typedef DataType Type;
			typedef DataType IntegralType;
		};
		template <> struct TypeTraits<float>
		{
			typedef float Type;
			typedef double IntegralType;
		};
		template <> struct TypeTraits<char>
		{
			typedef char Type;
			typedef int IntegralType;
		};
		template <> struct TypeTraits<short>
		{
			typedef short Type;
			typedef long IntegralType;
		};
		template <> struct TypeTraits<unsigned short>
		{
			typedef unsigned short Type;
			typedef unsigned long IntegralType;
		};
		template <> struct TypeTraits<unsigned char>
		{
			typedef unsigned char Type;
			typedef unsigned int IntegralType;
		};
		template <> struct TypeTraits<int>
		{
			typedef int Type;
			typedef long IntegralType;
		};
		template <> struct TypeTraits<unsigned int>
		{
			typedef unsigned int Type;
			typedef unsigned long IntegralType;
		};

		/** \brief Determines an integral image representation for a given organized data array
		  * \author Suat Gedikli
		  */

		template<typename T> bool isFinite(T arg)
		{
			return arg == arg &&
				arg != std::numeric_limits<T>::infinity() &&
				arg != -std::numeric_limits<T>::infinity();
		}

		template <class DataType, unsigned Dimension> class IntegralImage2D
		{
		public:
			static const unsigned dim_fst = Dimension;
			static const unsigned dim_snd = (Dimension * (Dimension + 1)) >> 1;
			typedef cv::Vec<typename TypeTraits<DataType>::IntegralType, dim_fst> FirstType;
			typedef cv::Vec<typename TypeTraits<DataType>::IntegralType, dim_snd> SecondType;


		private:
			typedef cv::Vec<typename TypeTraits<DataType>::Type, dim_fst> InputType;

			std::vector<FirstType>  m_img_fst;
			std::vector<SecondType> m_img_snd;
			std::vector<unsigned>   m_img_fin;

			/** \brief The height of the 2d input data array */
			unsigned m_height;

			/** \brief The width of the 2d input data array */
			unsigned m_width;

			/** \brief Indicates whether second order integral images are available *  */
			bool m_com_snd;


		public:
			/** \brief Constructor for an Integral Image
			  * \param[in] snd set to true if we want to compute a second order image
			  */
			IntegralImage2D(bool snd = false) : m_height(1), m_width(1), m_com_snd(snd) { }

			/** \brief Destructor */
			virtual ~IntegralImage2D() { }

			/** \brief Set the input data to compute the integral image for
			  * \param[in] dat the input data
			  * \param[in] wdt the width of the data
			  * \param[in] hgt the height of the data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void setInput(const DataType* dat, unsigned wdt, unsigned hgt, unsigned srd_ele, unsigned srd_row)
			{
				if ((wdt + 1) * (hgt + 1) > m_img_fst.size())
				{
					this->m_height = hgt;
					this->m_width = wdt;

					m_img_fst.resize((wdt + 1) * (hgt + 1));
					m_img_fin.resize((wdt + 1) * (hgt + 1));
					if (m_com_snd)
						m_img_snd.resize((wdt + 1) * (hgt + 1));
				}
				computeIntegralImages(dat, srd_ele, srd_row);
			}
			/** \brief Set the input data to compute the integral image for (with a mask consideration)
			  * \param[in] dat the input data
			  * \param[in] msk the mask data (0 invalide data, else valide data)
			  * \param[in] wdt the width of the data
			  * \param[in] hgt the height of the data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void setInput(const DataType* dat, const uchar* msk, unsigned wdt, unsigned hgt, unsigned srd_ele, unsigned srd_row)
			{
				if ((wdt + 1) * (hgt + 1) > m_img_fst.size())
				{
					this->m_height = hgt;
					this->m_width = wdt;

					m_img_fst.resize((wdt + 1) * (hgt + 1));
					m_img_fin.resize((wdt + 1) * (hgt + 1));
					if (m_com_snd)
						m_img_snd.resize((wdt + 1) * (hgt + 1));
				}
				computeIntegralImages(dat, msk, srd_ele, srd_row);
			}
			/** \brief sets the computation for second order integral images on or off.
			  */
			void setSecondOrderComputation(bool in_com_snd) { m_com_snd = in_com_snd; }


			/** \brief Compute the first order sum within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline FirstType getFirstOrderSum(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_fst[lower_right_idx] + m_img_fst[upper_left_idx] - m_img_fst[upper_right_idx] - m_img_fst[lower_left_idx]);
			}
			/** \brief Compute the second order sum within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline SecondType getSecondOrderSum(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_snd[lower_right_idx] + m_img_snd[upper_left_idx] - m_img_snd[upper_right_idx] - m_img_snd[lower_left_idx]);
			}
			/** \brief Compute the number of finite elements within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline unsigned getFiniteElementsCount(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_fin[lower_right_idx] + m_img_fin[upper_left_idx] - m_img_fin[upper_right_idx] - m_img_fin[lower_left_idx]);
			}

			/** \brief Compute the first order sum within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline FirstType getFirstOrderSumSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_fst[lower_right_idx] + m_img_fst[upper_left_idx] - m_img_fst[upper_right_idx] - m_img_fst[lower_left_idx]);
			}
			/** \brief Compute the second order sum within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline SecondType getSecondOrderSumSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_snd[lower_right_idx] + m_img_snd[upper_left_idx] - m_img_snd[upper_right_idx] - m_img_snd[lower_left_idx]);
			}
			/** \brief Compute the number of finite elements within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline unsigned getFiniteElementsCountSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_fin[lower_right_idx] + m_img_fin[upper_left_idx] - m_img_fin[upper_right_idx] - m_img_fin[lower_left_idx]);
			}


		private:
			/** \brief Compute the actual integral image data
			  * \param[in] dat the input data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void computeIntegralImages(const DataType* dat, unsigned srd_ele, unsigned srd_row)
			{
				FirstType* previous_row = &m_img_fst[0];
				FirstType* current_row = previous_row + (m_width + 1);
				memset(previous_row, 0, sizeof(FirstType) * (m_width + 1));

				unsigned* count_previous_row = &m_img_fin[0];
				unsigned* count_current_row = count_previous_row + (m_width + 1);
				memset(count_previous_row, 0, sizeof(unsigned) * (m_width + 1));

				if (!m_com_snd)
				{
					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = current_row[0].all(0);
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							const InputType* element = reinterpret_cast<const InputType*>(&dat[val_idx]);
							if (isFinite(cv::sum(*element)[0]))
							{
								current_row[col_idx + 1] += static_cast<FirstType>(*element);
								++(count_current_row[col_idx + 1]);
							}

							//TESTER
							//float dat[12]
							//const Vec3f* elem = reinterpret_cast<const Vec3f*>(&dat[4]);
							//cout << "Elem: " << elem->val[0] << ", " << elem->val[1] << ", " << elem->val[2] << endl;
							//float scal = cv::sum(*elem)[0]; cout << "Sum: " << scal << endl;
							//Vec3d tot(2.5, 3.0, 3.5);
							//tot += static_cast<Vec3d>(*elem);

							val_idx += srd_ele;
						}
						dat += srd_row;
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
					}
				}
				else
				{
					SecondType* so_previous_row = &m_img_snd[0];
					SecondType* so_current_row = so_previous_row + (m_width + 1);
					memset(so_previous_row, 0, sizeof(SecondType) * (m_width + 1));

					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = current_row[0].all(0);
						so_current_row[0] = so_current_row[0].all(0);
						count_current_row[0] = 0;
						for (unsigned col_idx = 0, val_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							so_current_row[col_idx + 1] = so_previous_row[col_idx + 1] + so_current_row[col_idx] - so_previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							const InputType* element = reinterpret_cast<const InputType*>(&dat[val_idx]);
							if (isFinite(cv::sum(*element)[0]))
							{
								current_row[col_idx + 1] += static_cast<FirstType>(*element);
								++(count_current_row[col_idx + 1]);

								for (unsigned my_idx = 0, el_idx = 0; my_idx < Dimension; my_idx++)
									for (unsigned mx_idx = my_idx; mx_idx < Dimension; mx_idx++, el_idx++)
										so_current_row[col_idx + 1][el_idx] += (*element)[my_idx] * (*element)[mx_idx];
							}
							val_idx += srd_ele;
						}

						dat += srd_row;
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
						so_previous_row = so_current_row;
						so_current_row += (m_width + 1);
					}
				}
			}
			/** \brief Compute the actual integral image data (with a mask consideration)
			  * \param[in] dat the input data
			  * \param[in] msk the mask data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void computeIntegralImages(const DataType* dat, const uchar* msk, unsigned srd_ele, unsigned srd_row)
			{
				FirstType* previous_row = &m_img_fst[0];
				FirstType* current_row = previous_row + (m_width + 1);
				memset(previous_row, 0, sizeof(FirstType) * (m_width + 1));

				unsigned* count_previous_row = &m_img_fin[0];
				unsigned* count_current_row = count_previous_row + (m_width + 1);
				memset(count_previous_row, 0, sizeof(unsigned) * (m_width + 1));

				if (!m_com_snd)
				{
					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = current_row[0].all(0);
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0, msk_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							const InputType* element = reinterpret_cast<const InputType*>(&dat[val_idx]);
							if (msk[msk_idx] == 255)
							{
								if (isFinite(cv::sum(*element)[0]))
								{
									current_row[col_idx + 1] += static_cast<FirstType>(*element);
									count_current_row[col_idx + 1] += 1;
								}
							}

							val_idx += srd_ele;
							msk_idx += 1;
						}
						dat += srd_row;
						msk += (m_width);
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
					}
				}
				else
				{
					SecondType* so_previous_row = &m_img_snd[0];
					SecondType* so_current_row = so_previous_row + (m_width + 1);
					memset(so_previous_row, 0, sizeof(SecondType) * (m_width + 1));

					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = current_row[0].all(0);
						so_current_row[0] = so_current_row[0].all(0);
						count_current_row[0] = 0;
						for (unsigned col_idx = 0, val_idx = 0, msk_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							so_current_row[col_idx + 1] = so_previous_row[col_idx + 1] + so_current_row[col_idx] - so_previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							const InputType* element = reinterpret_cast<const InputType*>(&dat[val_idx]);
							if (msk[msk_idx] == 255)
							{
								if (isFinite(cv::sum(*element)[0]))
								{
									current_row[col_idx + 1] += static_cast<FirstType>(*element);
									count_current_row[col_idx + 1] += 1;

									for (unsigned my_idx = 0, el_idx = 0; my_idx < Dimension; my_idx++)
										for (unsigned mx_idx = my_idx; mx_idx < Dimension; mx_idx++, el_idx++)
											so_current_row[col_idx + 1][el_idx] += (*element)[my_idx] * (*element)[mx_idx];
								}
							}

							val_idx += srd_ele;
							msk_idx += 1;
						}

						dat += srd_row;
						msk += (m_width);
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
						so_previous_row = so_current_row;
						so_current_row += (m_width + 1);
					}
				}
			}
		};

		/**
		* \brief partial template specialization for integral images with just one channel.
		  */
		template <class DataType> class IntegralImage2D <DataType, 1>
		{
		public:
			static const unsigned dim_fst = 1;
			static const unsigned dim_snd = 1;
			typedef typename TypeTraits<DataType>::IntegralType FirstType;
			typedef typename TypeTraits<DataType>::IntegralType SecondType;


		private:
			std::vector<FirstType>  m_img_fst;
			std::vector<SecondType> m_img_snd;
			std::vector<unsigned>   m_img_fin;

			/** \brief The height of the 2d input data array */
			unsigned m_height;
			/** \brief The width of the 2d input data array */
			unsigned m_width;

			/** \brief Indicates whether second order integral images are available *  */
			bool m_com_snd;


		public:
			/** \brief Constructor for an Integral Image
			  * \param[in] snd set to true if we want to compute a second order image
			  */
			IntegralImage2D(bool snd = false) : m_height(1), m_width(1), m_com_snd(snd) { }

			/** \brief Destructor */
			virtual ~IntegralImage2D() { }

			/** \brief Set the input data to compute the integral image for
			  * \param[in] dat the input data
			  * \param[in] wdt the width of the data
			  * \param[in] hgt the height of the data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void setInput(const DataType* dat, unsigned wdt, unsigned hgt, unsigned srd_ele, unsigned srd_row)
			{
				if ((wdt + 1) * (hgt + 1) > m_img_fst.size())
				{
					this->m_height = hgt;
					this->m_width = wdt;

					m_img_fst.resize((wdt + 1) * (hgt + 1));
					m_img_fin.resize((wdt + 1) * (hgt + 1));
					if (m_com_snd)
						m_img_snd.resize((wdt + 1) * (hgt + 1));
				}
				computeIntegralImages(dat, srd_ele, srd_row);
			}
			/** \brief Set the input data to compute the integral image for (with a mask consideration)
			  * \param[in] dat the input data
			  * \param[in] msk the mask data
			  * \param[in] wdt the width of the data
			  * \param[in] hgt the height of the data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void setInput(const DataType* dat, const uchar* msk, unsigned wdt, unsigned hgt, unsigned srd_ele, unsigned srd_row)
			{
				if ((wdt + 1) * (hgt + 1) > m_img_fst.size())
				{
					this->m_height = hgt;
					this->m_width = wdt;

					m_img_fst.resize((wdt + 1) * (hgt + 1));
					m_img_fin.resize((wdt + 1) * (hgt + 1));
					if (m_com_snd)
						m_img_snd.resize((wdt + 1) * (hgt + 1));
				}
				computeIntegralImages(dat, msk, srd_ele, srd_row);
			}


			/** \brief sets the computation for second order integral images on or off.
			  */
			void setSecondOrderComputation(bool in_com_snd) { m_com_snd = in_com_snd; }


			/** \brief Compute the first order sum within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline FirstType getFirstOrderSum(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_fst[lower_right_idx] + m_img_fst[upper_left_idx] - m_img_fst[upper_right_idx] - m_img_fst[lower_left_idx]);
			}
			/** \brief Compute the second order sum within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline SecondType getSecondOrderSum(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_snd[lower_right_idx] + m_img_snd[upper_left_idx] - m_img_snd[upper_right_idx] - m_img_snd[lower_left_idx]);
			}
			/** \brief Compute the number of finite elements within a given rectangle
			  * \param[in] start_x x position of rectangle
			  * \param[in] start_y y position of rectangle
			  * \param[in] width width of rectangle
			  * \param[in] height height of rectangle
			  */
			inline unsigned getFiniteElementsCount(unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = upper_left_idx + width;
				const unsigned lower_left_idx = (start_y + height) * (m_width + 1) + start_x;
				const unsigned lower_right_idx = lower_left_idx + width;

				return(m_img_fin[lower_right_idx] + m_img_fin[upper_left_idx] - m_img_fin[upper_right_idx] - m_img_fin[lower_left_idx]);
			}

			/** \brief Compute the first order sum within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline FirstType getFirstOrderSumSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_fst[lower_right_idx] + m_img_fst[upper_left_idx] - m_img_fst[upper_right_idx] - m_img_fst[lower_left_idx]);
			}
			/** \brief Compute the second order sum within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline SecondType getSecondOrderSumSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_snd[lower_right_idx] + m_img_snd[upper_left_idx] - m_img_snd[upper_right_idx] - m_img_snd[lower_left_idx]);
			}
			/** \brief Compute the number of finite elements within a given rectangle
			  * \param[in] start_x x position of the start of the rectangle
			  * \param[in] start_y x position of the start of the rectangle
			  * \param[in] end_x x position of the end of the rectangle
			  * \param[in] end_y x position of the end of the rectangle
			  */
			inline unsigned getFiniteElementsCountSE(unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y)
			{
				const unsigned upper_left_idx = start_y * (m_width + 1) + start_x;
				const unsigned upper_right_idx = start_y * (m_width + 1) + end_x;
				const unsigned lower_left_idx = end_y * (m_width + 1) + start_x;
				const unsigned lower_right_idx = end_y * (m_width + 1) + end_x;

				return(m_img_fin[lower_right_idx] + m_img_fin[upper_left_idx] - m_img_fin[upper_right_idx] - m_img_fin[lower_left_idx]);
			}


		private:
			/** \brief Compute the actual integral image data
			  * \param[in] dat the input data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void computeIntegralImages(const DataType* dat, unsigned srd_ele, unsigned srd_row)
			{
				FirstType* previous_row = &m_img_fst[0];
				FirstType* current_row = previous_row + (m_width + 1);
				memset(previous_row, 0, sizeof(FirstType) *(m_width + 1));

				unsigned* count_previous_row = &m_img_fin[0];
				unsigned* count_current_row = count_previous_row + (m_width + 1);
				memset(count_previous_row, 0, sizeof(unsigned) * (m_width + 1));

				if (!m_com_snd)
				{
					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = 0;
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							if (isFinite(dat[val_idx]))
							{
								current_row[col_idx + 1] += dat[val_idx];
								count_current_row[col_idx + 1] += 1;
							}
							val_idx += srd_ele;
						}
						dat += srd_row;
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
					}
				}
				else
				{
					SecondType* so_previous_row = &m_img_snd[0];
					SecondType* so_current_row = so_previous_row + (m_width + 1);
					memset(so_previous_row, 0, sizeof(SecondType) * (m_width + 1));

					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = 0;
						so_current_row[0] = 0;
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							so_current_row[col_idx + 1] = so_previous_row[col_idx + 1] + so_current_row[col_idx] - so_previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							if (isFinite(dat[val_idx]))
							{
								current_row[col_idx + 1] += dat[val_idx];
								so_current_row[col_idx + 1] += dat[val_idx] * dat[val_idx];
								++(count_current_row[col_idx + 1]);
							}
							val_idx += srd_ele;
						}
						dat += srd_row;
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
						so_previous_row = so_current_row;
						so_current_row += (m_width + 1);
					}
				}
			}
			/** \brief Compute the actual integral image data (with a mask consideration)
			  * \param[in] dat the input data
			  * \param[in] msk the mask data
			  * \param[in] srd_ele the element stride of the data
			  * \param[in] srd_row the row stride of the data
			  */
			void computeIntegralImages(const DataType* dat, const uchar* msk, unsigned srd_ele, unsigned srd_row)
			{
				FirstType* previous_row = &m_img_fst[0];
				FirstType* current_row = previous_row + (m_width + 1);
				memset(previous_row, 0, sizeof(FirstType) * (m_width + 1));

				unsigned* count_previous_row = &m_img_fin[0];
				unsigned* count_current_row = count_previous_row + (m_width + 1);
				memset(count_previous_row, 0, sizeof(unsigned) * (m_width + 1));

				if (!m_com_snd)
				{
					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = FirstType(0.0);
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0, msk_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							if (msk[msk_idx] == 255)
							{
								if (isFinite(dat[val_idx]))
								{
									current_row[col_idx + 1] += dat[val_idx];
									count_current_row[col_idx + 1] += 1;
								}
							}

							val_idx += srd_ele;
							msk_idx += 1;
						}
						dat += srd_row;
						msk += (m_width);
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
					}
				}
				else
				{
					SecondType* so_previous_row = &m_img_snd[0];
					SecondType* so_current_row = so_previous_row + (m_width + 1);
					memset(so_previous_row, 0, sizeof(SecondType) * (m_width + 1));

					for (unsigned row_idx = 0; row_idx < m_height; row_idx++)
					{
						current_row[0] = FirstType(0.0);
						so_current_row[0] = SecondType(0.0);
						count_current_row[0] = 0;

						for (unsigned col_idx = 0, val_idx = 0, msk_idx = 0; col_idx < m_width; col_idx++)
						{
							current_row[col_idx + 1] = previous_row[col_idx + 1] + current_row[col_idx] - previous_row[col_idx];
							so_current_row[col_idx + 1] = so_previous_row[col_idx + 1] + so_current_row[col_idx] - so_previous_row[col_idx];
							count_current_row[col_idx + 1] = count_previous_row[col_idx + 1] + count_current_row[col_idx] - count_previous_row[col_idx];

							if (msk[msk_idx] == 255)
							{
								if (isFinite(dat[val_idx]))
								{
									current_row[col_idx + 1] += dat[val_idx];
									so_current_row[col_idx + 1] += dat[val_idx] * dat[val_idx];
									count_current_row[col_idx + 1] += 1;
								}
							}

							val_idx += srd_ele;
							msk_idx += 1;
						}

						dat += srd_row;
						msk += (m_width);
						previous_row = current_row;
						current_row += (m_width + 1);
						count_previous_row = count_current_row;
						count_current_row += (m_width + 1);
						so_previous_row = so_current_row;
						so_current_row += (m_width + 1);
					}
				}
			}

		};
	}
}

#endif // INTEGRALIMAGE2D_H
