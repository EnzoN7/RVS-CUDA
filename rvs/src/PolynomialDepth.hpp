#ifndef _POLYNOMIALDEPTH_HPP_
#define _POLYNOMIALDEPTH_HPP_

#include <opencv2/core.hpp>

namespace rvs
{
	/**
	Class representing an Bezier depth map
	*/
	class PolynomialDepth
	{
        public:

		/** Initialize all maps at once */
		PolynomialDepth();
		PolynomialDepth (std::array<cv::Mat1f,20> polynomial);

		~PolynomialDepth();

        std::array<cv::Mat1f,20> m_polynomial;
    };

}

#endif
