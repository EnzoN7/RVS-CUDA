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

#include "SynthesizedView.hpp"


namespace rvs
{
    template<typename position_t,
             typename color_t,
             typename channel_t>
	SynthesizedView<position_t, color_t, channel_t>::SynthesizedView() {}
    
    template<typename position_t,
             typename color_t,
             typename channel_t>
	SynthesizedView<position_t, color_t, channel_t>::~SynthesizedView() {}

    template<typename position_t,
             typename color_t,
             typename channel_t>
	SynthetizedViewTriangle<position_t, color_t, channel_t>::SynthetizedViewTriangle(cv::Size realSize, cv::Size virtualSize, SpaceTransformer const* object)
		: m_realSize(realSize)
		, m_virtualSize(virtualSize)
		, m_devNormalizedColor(nullptr)
		, m_devNormalizedDepth(nullptr)
		, m_space_transformer(object)
		, m_inputParams(m_space_transformer->getInputParameters())
        , m_wrappingMethod(WrappingMethod::none)
	{
		cudaMalloc(&m_devOutputDepthValidity, virtualSize.area() * sizeof(encoded_t<channel_t>));
		cudaMalloc(&m_devColorLock, virtualSize.area() * sizeof(int));

		cudaMalloc(&m_devTransformedPosition, realSize.area() * sizeof(position_t));
		cudaMalloc(&m_devTransformedDepth, realSize.area() * sizeof(channel_t));

		cudaMalloc(&m_devVirtualColor, virtualSize.area() * sizeof(color_t));
		cudaMalloc(&m_devVirtualDepth, virtualSize.area() * sizeof(channel_t));
		cudaMalloc(&m_devVirtualValidity, virtualSize.area() * sizeof(position_t));

		cudaEventCreateWithFlags(&m_synthesizeVirtualDepthValidity, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_initDepthValidity, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_colorizeTriangles, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&m_projection, cudaEventDisableTiming);
	}

    template<typename position_t,
             typename color_t,
             typename channel_t>
	SynthetizedViewTriangle<position_t, color_t, channel_t>::~SynthetizedViewTriangle()
	{
		cudaFree(m_devOutputDepthValidity);
		cudaFree(m_devColorLock);

		cudaFree(m_devTransformedPosition);
		cudaFree(m_devTransformedDepth);

		cudaFree(m_devVirtualColor);
		cudaFree(m_devVirtualDepth);
		cudaFree(m_devVirtualValidity);

		cudaEventDestroy(m_synthesizeVirtualDepthValidity);
		cudaEventDestroy(m_initDepthValidity);
		cudaEventDestroy(m_colorizeTriangles);
		cudaEventDestroy(m_projection);
	}

    template<typename position_t,
             typename color_t,
             typename channel_t>
	void SynthetizedViewTriangle<position_t, color_t, channel_t>::compute(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2)
	{
        m_futureProjection = std::async(std::launch::async,
            [this, &inputImage, &stream2]()
            {
                Parameters virtualParams = m_space_transformer->getVirtualParameters();
                //@Hope
                /*cv::Matx33f R = m_space_transformer->get_rotation();
                cv::Vec3f t = m_space_transformer->get_translation();
                prepareParameters(m_realSize, m_inputParams.getHorRange(), m_inputParams.getVerRange(), R, t, virtualParams.getFocal(), virtualParams.getPrinciplePoint());*/

                prepareParameters(m_realSize, m_inputParams, virtualParams);
                
                m_devNormalizedDepth = inputImage->waitAndGetNormalizedDepth();

                //@HoPe
                //When input views are perspective
                if(m_inputParams.getProjectionType()=="Perspective")
                    unprojectTo2D_projectTo2D(virtualParams, stream2); 
                else
                    unprojectTo3D_projectTo2D(virtualParams, stream2);

                if (virtualParams.getProjectionType() != "Equirectangular" && detail::g_rescale < 1.0f)
                    scaleUV(m_devTransformedPosition, detail::g_rescale, m_realSize.area(), stream2);

                cudaEventRecord(m_projection, stream2);
            }
        );

		synthesizeImage(inputImage, stream1, stream2);
	}

    template<typename position_t,
             typename color_t,
             typename channel_t>
	void SynthetizedViewTriangle<position_t, color_t, channel_t>::unprojectTo3D_projectTo2D(Parameters& virtualParams, cudaStream_t& stream)
	{

		if (virtualParams.getProjectionType() == "Equirectangular")
		{
			unprojectERP_projectERP<position_t, channel_t>(m_realSize,
									                       m_devNormalizedDepth,
									                       m_devTransformedPosition,
									                       m_devTransformedDepth,
									                       m_precomputedParams,
									                       stream);

			m_wrappingMethod = m_inputParams.isFullHorRange()
							 ? WrappingMethod::horizontal
							 : WrappingMethod::none;
		}
		else
		{
			unprojectERP_projectPerspective<position_t, channel_t>(m_realSize,
											                       m_devNormalizedDepth,
											                       m_devTransformedPosition,
											                       m_devTransformedDepth,
											                       m_precomputedParams,
											                       stream);
		}
	}


    //@HoPe
    //When input views are perspective
    template<typename position_t,
        typename color_t,
        typename channel_t>
    void SynthetizedViewTriangle<position_t, color_t, channel_t>::unprojectTo2D_projectTo2D(Parameters& virtualParams, cudaStream_t& stream)
    {
            unprojectPerspective_projectPerspective<position_t, channel_t>(m_realSize,
                m_devNormalizedDepth,
                m_devTransformedPosition,
                m_devTransformedDepth,
                m_precomputedParams,
                stream);
        
    }


    template<typename position_t,
             typename color_t,
             typename channel_t>
	void SynthetizedViewTriangle<position_t, color_t, channel_t>::synthesizeImage(std::shared_ptr<InputView<channel_t, color_t>>& inputImage, cudaStream_t& stream1, cudaStream_t& stream2)
	{
		bool wrapHorizontal = (m_wrappingMethod == WrappingMethod::horizontal);

        init<channel_t>(m_virtualSize, m_devColorLock, m_devOutputDepthValidity, stream1, stream2, m_initDepthValidity);

        m_devNormalizedColor = inputImage->waitAndGetNormalizedColor();

		synthesizeImageWithTrianglesMethod<position_t, color_t, channel_t>(m_devNormalizedColor,
										                                   m_realSize,
										                                   m_devTransformedDepth,
										                                   m_devTransformedPosition,
										                                   m_virtualSize,
										                                   wrapHorizontal,
										                                   m_devVirtualColor,
										                                   m_devVirtualDepth,
										                                   m_devVirtualValidity,
										                                   stream1, stream2,
										                                   m_devOutputDepthValidity, m_devColorLock,
										                                   m_synthesizeVirtualDepthValidity, m_initDepthValidity, m_colorizeTriangles, m_projection,
                                                                           m_futureProjection);
	}


    //@HoPe Comment) 
    //TODO: Remove this method later
    template<typename position_t,
             typename color_t,
             typename channel_t>
    void SynthetizedViewTriangle<position_t, color_t, channel_t>::prepareParameters(cv::Size size, cv::Vec2f hor_range, cv::Vec2f ver_range,
													cv::Matx33f R, cv::Vec3f t, cv::Vec2f f, cv::Vec2f p)
    {
        m_precomputedParams.camData.rotation[0] = R(0, 0);
        m_precomputedParams.camData.rotation[1] = R(0, 1);
        m_precomputedParams.camData.rotation[2] = R(0, 2);
        m_precomputedParams.camData.rotation[3] = R(1, 0);
        m_precomputedParams.camData.rotation[4] = R(1, 1);
        m_precomputedParams.camData.rotation[5] = R(1, 2);
        m_precomputedParams.camData.rotation[6] = R(2, 0);
        m_precomputedParams.camData.rotation[7] = R(2, 1);
        m_precomputedParams.camData.rotation[8] = R(2, 2);

        m_precomputedParams.camData.translation[0] = t[0];
        m_precomputedParams.camData.translation[1] = t[1];
        m_precomputedParams.camData.translation[2] = t[2];

        if (size != m_lastSize)
        {
            int blockWidth = 8;
            int blockHeight = 16;
            m_precomputedParams.blockDim = dim3(blockWidth, blockHeight);
            m_precomputedParams.gridDim = dim3((size.width - 1 + blockWidth) / blockWidth,
											   (size.height - 1 + blockHeight) / blockHeight);

            float radperdeg = 0.01745329252f;
            m_precomputedParams.dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;
            m_precomputedParams.dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

            m_lastSize = size;
        }

        if (hor_range != m_lastHorRange)
        {
            float radperdeg = 0.01745329252f;
            m_precomputedParams.devPhi0 = radperdeg * hor_range[1];
            m_precomputedParams.dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;


            if (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f)
            {
                const float degperrad = 57.295779513f;
                m_precomputedParams.devU0 = size.width * hor_range[1] / (hor_range[1] - hor_range[0]);
                m_precomputedParams.dev_du_dphi = -degperrad * size.width / (hor_range[1] - hor_range[0]);
            }
            m_lastHorRange = hor_range;
        }

        if (ver_range != m_lastVerRange)
        {
            float radperdeg = 0.01745329252f;
            m_precomputedParams.devTheta0 = radperdeg * ver_range[1];
            m_precomputedParams.dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

            if (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f)
            {
                const float degperrad = 57.295779513f;
                m_precomputedParams.devV0 = size.height * ver_range[1] / (ver_range[1] - ver_range[0]);
                m_precomputedParams.dev_dv_dtheta = -degperrad * size.height / (ver_range[1] - ver_range[0]);
            }
            m_lastVerRange = ver_range;
        }

        if (f != m_lastF)
        {
            m_precomputedParams.camData.focal[0] = f[0];
            m_precomputedParams.camData.focal[1] = f[1];
            if (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f)
            {
                const float degperrad = 57.295779513f;
                m_precomputedParams.devU0 = size.width * hor_range[1] / (hor_range[1] - hor_range[0]);
                m_precomputedParams.dev_du_dphi = -degperrad * size.width / (hor_range[1] - hor_range[0]);
            }
            else
            {
                m_precomputedParams.devU0 = 0;
                m_precomputedParams.dev_du_dphi = 0;
            }
            m_lastF = f;
        }

        if (p != m_lastP)
        {
            m_precomputedParams.camData.principlePoint[0] = p[0];
            m_precomputedParams.camData.principlePoint[1] = p[1];
            if (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f)
            {
                const float degperrad = 57.295779513f;
                m_precomputedParams.devV0 = size.height * ver_range[1] / (ver_range[1] - ver_range[0]);
                m_precomputedParams.dev_dv_dtheta = -degperrad * size.height / (ver_range[1] - ver_range[0]);
            }
            else
            {
                m_precomputedParams.devV0 = 0;
                m_precomputedParams.dev_dv_dtheta = 0;
            }
            m_lastP = p;
        }
    }


    //@HoPe  
    //Modified prepareParameters method that add input perspective view params to the PrecomputedParams
    template<typename position_t,
        typename color_t,
        typename channel_t>
void SynthetizedViewTriangle<position_t, color_t, channel_t>::prepareParameters(cv::Size size, const Parameters& input_params, Parameters& virtualParams)
{
      
    cv::Matx33f R = m_space_transformer->get_rotation();
    cv::Vec3f t = m_space_transformer->get_translation();

    cv::Vec2f hor_range = input_params.getHorRange();
    cv::Vec2f ver_range = input_params.getVerRange();

    cv::Vec2f f = virtualParams.getFocal(); 
    cv::Vec2f p = virtualParams.getPrinciplePoint();

    //When the output does not have Focal (f) and PrinciplePoint (p), it means that it is ERP
    //bool isVirtualERP =  (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f);
    //But we can only check projection type!
    bool isVirtualERP = (virtualParams.getProjectionType() == "Equirectangular");

    m_precomputedParams.camData.rotation[0] = R(0, 0);
    m_precomputedParams.camData.rotation[1] = R(0, 1);
    m_precomputedParams.camData.rotation[2] = R(0, 2);
    m_precomputedParams.camData.rotation[3] = R(1, 0);
    m_precomputedParams.camData.rotation[4] = R(1, 1);
    m_precomputedParams.camData.rotation[5] = R(1, 2);
    m_precomputedParams.camData.rotation[6] = R(2, 0);
    m_precomputedParams.camData.rotation[7] = R(2, 1);
    m_precomputedParams.camData.rotation[8] = R(2, 2);

    m_precomputedParams.camData.translation[0] = t[0];
    m_precomputedParams.camData.translation[1] = t[1];
    m_precomputedParams.camData.translation[2] = t[2];

    //For input ERP these are always zero
    cv::Vec2f in_f=  input_params.getFocal();
    cv::Vec2f in_p = input_params.getPrinciplePoint();
    m_precomputedParams.in_focal[0] = in_f[0];
    m_precomputedParams.in_focal[1] = in_f[1];
    m_precomputedParams.in_principlePoint[0] = in_p[0];
    m_precomputedParams.in_principlePoint[1] = in_p[1];

    if (size != m_lastSize)
    {
        int blockWidth = 8;
        int blockHeight = 16;
        m_precomputedParams.blockDim = dim3(blockWidth, blockHeight);
        m_precomputedParams.gridDim = dim3((size.width - 1 + blockWidth) / blockWidth,
            (size.height - 1 + blockHeight) / blockHeight);

        float radperdeg = 0.01745329252f;
        m_precomputedParams.dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;
        m_precomputedParams.dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

        m_lastSize = size;
    }

    if (hor_range != m_lastHorRange)
    {
        float radperdeg = 0.01745329252f;
        m_precomputedParams.devPhi0 = radperdeg * hor_range[1];
        m_precomputedParams.dev_dphi_du = -radperdeg * (hor_range[1] - hor_range[0]) / size.width;

       
        if (isVirtualERP)
        {
            const float degperrad = 57.295779513f;
            m_precomputedParams.devU0 = size.width * hor_range[1] / (hor_range[1] - hor_range[0]);
            m_precomputedParams.dev_du_dphi = -degperrad * size.width / (hor_range[1] - hor_range[0]);
        }
        m_lastHorRange = hor_range;
    }

    if (ver_range != m_lastVerRange)
    {
        float radperdeg = 0.01745329252f;
        m_precomputedParams.devTheta0 = radperdeg * ver_range[1];
        m_precomputedParams.dev_dtheta_dv = -radperdeg * (ver_range[1] - ver_range[0]) / size.height;

        if (isVirtualERP)
        {
            const float degperrad = 57.295779513f;
            m_precomputedParams.devV0 = size.height * ver_range[1] / (ver_range[1] - ver_range[0]);
            m_precomputedParams.dev_dv_dtheta = -degperrad * size.height / (ver_range[1] - ver_range[0]);
        }
        m_lastVerRange = ver_range;
    }

    if (f != m_lastF)
    {
        m_precomputedParams.camData.focal[0] = f[0];
        m_precomputedParams.camData.focal[1] = f[1];
        if (isVirtualERP)
        {
            const float degperrad = 57.295779513f;
            m_precomputedParams.devU0 = size.width * hor_range[1] / (hor_range[1] - hor_range[0]);
            m_precomputedParams.dev_du_dphi = -degperrad * size.width / (hor_range[1] - hor_range[0]);
        }
        else
        {
            m_precomputedParams.devU0 = 0;
            m_precomputedParams.dev_du_dphi = 0;
        }
        m_lastF = f;
    }

    if (p != m_lastP)
    {
        m_precomputedParams.camData.principlePoint[0] = p[0];
        m_precomputedParams.camData.principlePoint[1] = p[1];
        if (f[0] == 0.f && f[1] == 0.f && p[0] == 0.f && p[1] == 0.f)
        {
            const float degperrad = 57.295779513f;
            m_precomputedParams.devV0 = size.height * ver_range[1] / (ver_range[1] - ver_range[0]);
            m_precomputedParams.dev_dv_dtheta = -degperrad * size.height / (ver_range[1] - ver_range[0]);
        }
        else
        {
            m_precomputedParams.devV0 = 0;
            m_precomputedParams.dev_dv_dtheta = 0;
        }
        m_lastP = p;
    }
}




    template class SynthesizedView<float2, float3, float>;
    template class SynthesizedView<double2, double3, double>;
    template class SynthesizedView<half2, half3, half>;

    template class SynthetizedViewTriangle<float2, float3, float>;
    template class SynthetizedViewTriangle<double2, double3, double>;
    template class SynthetizedViewTriangle<half2, half3, half>;
}
