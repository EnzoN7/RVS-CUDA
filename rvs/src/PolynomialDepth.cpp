#include "PolynomialDepth.hpp"
#include "JsonParser.hpp"
#include <iostream>


namespace rvs{
    PolynomialDepth::PolynomialDepth(){}
    PolynomialDepth::PolynomialDepth(std::array<cv::Mat1f,20> polynomial){
        m_polynomial = polynomial;
    }
    PolynomialDepth::~PolynomialDepth(){}
}
