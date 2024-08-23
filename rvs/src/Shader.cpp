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

#include "Shader.hpp"

#include <cassert>

namespace rvs
{
	namespace opengl
	{
		void Shader::shader_compile_errors(const GLuint &object, const char * type) {
			std::string type_(type);
			GLint success;
			GLchar infoLog[1024];

			bool error = false;

			if (type_ != "Program") {
				glGetShaderiv(object, GL_COMPILE_STATUS, &success);
				if (!success) {
					glGetShaderInfoLog(object, 1024, NULL, infoLog);
					printf("| ERROR::SHADER: Compile-time error -> Type: %s\n", type);
					error = true;
				}
			}
			else {
				glGetProgramiv(object, GL_LINK_STATUS, &success);
				if (!success) {
					glGetProgramInfoLog(object, 1024, NULL, infoLog);
					printf("| ERROR::Shader: Link-time error -> Type: %s\n", type);
					error = true;
				}
			}

			if (error) {
				printf("%s\n", infoLog);
				printf("-- --------------------------------------------------- -- \n");
			}
		}

		const char * Shader::shader_type2string(GLenum shaderType) {
			std::string ret;

			switch (shaderType) {
			case GL_COMPUTE_SHADER:
				ret = "Compute";
				break;
			case GL_VERTEX_SHADER:
				ret = "Vertex";
				break;
			case GL_TESS_CONTROL_SHADER:
				ret = "Tessellation Control";
				break;
			case GL_TESS_EVALUATION_SHADER:
				ret = "Tessellation Evaluation";
				break;
			case GL_GEOMETRY_SHADER:
				ret = "Geometry";
				break;
			case GL_FRAGMENT_SHADER:
				ret = "Fragment";
				break;
			default:
				ret = "unknown";
				break;
			}
			return ret.c_str();
		}

		GLuint Shader::compile_shader(const std::string shader, GLenum shaderType)
		{
			const GLchar * ccode = shader.c_str();

			const GLuint id = glCreateShader(shaderType);
			glShaderSource(id, 1, &ccode, NULL);
			glCompileShader(id);
			shader_compile_errors(id, shader_type2string(shaderType));

			return id;
		}

		Shader::Shader(std::string vertexSource, std::string fragmentSource, std::string geometrySource)
			: m_ID(0)
		{
			// Compile shaders
			assert(!vertexSource.empty() && !fragmentSource.empty());
			auto vertexShader = compile_shader(vertexSource, GL_VERTEX_SHADER);
			auto fragmentShader = compile_shader(fragmentSource, GL_FRAGMENT_SHADER);
			auto geometryShader = geometrySource.empty() ? GLuint(0) : compile_shader(geometrySource, GL_GEOMETRY_SHADER);

			// Attach shaders to a new program
			auto ID = glCreateProgram();
			glAttachShader(ID, vertexShader);
			glAttachShader(ID, fragmentShader);
			if (geometryShader) {
				glAttachShader(ID, geometryShader);
			}
			glLinkProgram(ID);
			shader_compile_errors(ID, "Program");
			m_ID = ID;
			printf("NEW PROGRAM WITH ID: %i\n", m_ID);

			// Delete the shaders as they're linked into our ID now and no longer necessery
			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);
			if (geometryShader) {
				glDeleteShader(geometryShader);
			}
		}

		GLuint Shader::getProgramID() const
		{
			return m_ID;
		}

		ShadersList::ShadersList()
			: m_shaders({
				{ "synthesis", Shader(
					getSynthesisVertexShaderSource(),
					getSynthesisFragmentShaderSource(),
					getSynthesisGeometryShaderSource())
				},
				{ "synthesis_polynomial", Shader(
					getSynthesisVertexShaderSourcePolynomial(),
					getSynthesisFragmentShaderSource(),
					getSynthesisGeometryShaderSource())
				},
				{ "blending1", Shader(
					getBlendingVertexShaderSource(),
					getBlendingFragmentShaderSource(1))
				},
				{ "blending2", Shader(
					getBlendingVertexShaderSource(),
					getBlendingFragmentShaderSource(2))
				}
				})
		{}

		ShadersList const& ShadersList::getInstance()
		{
			static ShadersList singleton;
			return singleton;
		}

		Shader const& ShadersList::operator () (const char* name) const
		{
			return m_shaders.at(name);
		}

		std::string ShadersList::getSynthesisVertexShaderSource()
		{
			return R"(
		#version 420 core

		layout(location = 0) in float empty_value;
		uniform sampler2D depth_texture;

		out VS_OUT {
			vec2 uv;
			float depth;
			vec3 pos_in; //for normal computation
			vec3 pos_out; //for normal computation
		} vs_out;

		uniform float w;
		uniform float h;
		uniform float n_w;
		uniform float n_h;
		uniform float sensor;


		// Rotation and translation
		uniform mat3 R;
		uniform vec3 t;

		// Input/output projection types
		uniform int erp_in;
		uniform int erp_out;

		// Perspective unprojection
		uniform vec2 f;
		uniform vec2 p;

		// Perspective projection
		uniform vec2 n_f;
		uniform vec2 n_p;

		// Equirectangular unprojection
		uniform float phi0;
		uniform float theta0;
		uniform float dphi_du;
		uniform float dtheta_dv;

		// Equirectangular projection
		uniform float u0;
		uniform float v0;
		uniform float du_dphi;
		uniform float dv_dtheta;

		uniform float max_depth;

		vec2 get_position_from_Vertex_ID(int id, float width) {
			int y = id / int(width);
			int x = id - y * int(width);
			return vec2(float(x), float(y));
		}

		vec3 unproject_equirectangular() {
			// Image coordinates
			vec2 xy = get_position_from_Vertex_ID(gl_VertexID, w);
			float d = texture(depth_texture, vec2(xy.x/w, xy.y/h)).x*max_depth;

			// Spherical coordinates
			float phi = phi0 + xy.x * dphi_du;
			float theta = theta0 + xy.y * dtheta_dv;

			// World coordinates
			return d * vec3(cos(theta) * cos(phi),
							cos(theta) * sin(phi),
							sin(theta));
		}

		vec3 unproject_perspective() {
			// Image coordinates
			vec2 xy = get_position_from_Vertex_ID(gl_VertexID, w);
			float d = texture(depth_texture, vec2(xy.x/w, xy.y/h)).x*max_depth;

			// World coordinates
			return vec3(
				d,
				-(d / f.x) * (xy.x - p.x),
				-(d / f.y) * (xy.y - p.y));
		}

		void project_equirectangular(vec3 r) {
			// Image coordinates
			vec2 xy = get_position_from_Vertex_ID(gl_VertexID, w);

			// Spherical coordinates
			float n = length(r);
			float phi = atan(r.y, r.x);
			float theta = asin(r.z / n);

			// Image coordinates in [-1, 1]
			float hic = u0 + du_dphi * phi;
			float vic = v0 + dv_dtheta * theta;

			vs_out.uv = vec2(xy.x / w, xy.y / h);
			gl_Position = vec4(hic, vic, 0., 1.);
			vs_out.depth = n;
		}

		void project_perspective(vec3 r) {
			// Image coordinates
			vec2 xy = get_position_from_Vertex_ID(gl_VertexID, w);

			vec2 uv = vec2(
				-n_f.x * r.y / r.x + n_p.x,
				-n_f.y * r.z / r.x + n_p.y);

			// Image coordinates (in [-1, 1])
			/*gl_Position = vec4(
				 2. * uv.x / w - 1.,
				-2. * uv.y / h + 1., 0., 1.);

			vs_out.uv = vec2(xy.x / w, xy.y / h);
			vs_out.depth = 1.;*/

			// Image coordinates (in [-1, 1])
			if(r.x > 0){
				gl_Position = vec4(2.0*uv.x/n_w-1.0, -2.0*uv.y/n_h+1.0, 1.f, 1.f);
			}
			else {
				gl_Position = vec4(
					 2. * uv.x / n_w - 1.,
					-2. * uv.y / n_h + 1., -1.0, 1.);
			}
			vs_out.uv = vec2(xy.x / w, xy.y / h);
			vs_out.depth = r.x;
		}

		void main(void)
		{
			vec3 eucl;

			if (erp_in == 1)
				eucl = unproject_equirectangular();
			else
				eucl = unproject_perspective();
			vs_out.pos_in = eucl;

			eucl = R * eucl + t;

			vs_out.pos_out = eucl;
			if (erp_out == 1)
				project_equirectangular(eucl);
			else
				project_perspective(eucl);
		}
	)";
		}

		std::string ShadersList::getSynthesisFragmentShaderSource()
		{
			return R"(
		#version 420 core

		uniform sampler2D image_texture;
		uniform sampler2D depth_texture;

		in vec2 gs_uv;
		in float gs_quality;
		in float gs_depth;

		layout(location=0) out vec3 color;
		layout(location=1) out float depth;
		layout(location=2) out float quality;

		uniform float max_depth;

		void main(void)
		{
			color = texture(image_texture, gs_uv).rgb;
			//depth = gs_depth/max_depth/1000.0;//*100.0;
			//quality = 0.01*gs_quality/depth/depth/depth; //version lambertien
			depth = gs_depth/max_depth;
			quality = gs_quality/15.0/depth/depth; 
 			//quality = 0.1*gs_quality; //version non-lambertien only

            if(quality >0.0f){
                gl_FragDepth = gs_depth/max_depth;
                depth = 1.0/gs_quality;}
                else
                {
                   gl_FragDepth = 0.99;
				   depth = 1.0;
				   quality = 0.00;
                } //version non-lambertien only
			return;
			
		}

	)";
		}

		std::string ShadersList::getSynthesisGeometryShaderSource()
		{
			return R"(
		#version 420 core

		layout(triangles) in;
		layout(triangle_strip, max_vertices = 3) out;

		in VS_OUT {
			vec2 uv;
			float depth;
			vec3 pos_in; //for normal computation
			vec3 pos_out; //for normal computation
		} gs_in[];

		out vec2 gs_uv;
		out float gs_quality;
		out float gs_depth;

		uniform float w;
		uniform float h;
		uniform float max_depth;
		uniform float min_depth;

		float get_quality() { 
			vec2 A = (gl_in[0].gl_Position).xy * vec2(w, h);
			vec2 B = (gl_in[1].gl_Position).xy * vec2(w, h);
			vec2 C = (gl_in[2].gl_Position).xy * vec2(w, h); //Coordinate in [-w,w]x[-h,h]

			float dab = length(A - B)/2.0;
			float dac = length(A - C)/2.0;
			float dbc = length(B - C)/2.0;

			float stretch = max(dbc, max(dab, dac));

			float quality = max(0.0,15.0 - stretch);
			return quality;
		}
		void gen_vertex(int index, float quality) {
			gs_quality = quality;
			if (quality > 000){
				gs_uv = gs_in[index].uv;
				gs_depth = gs_in[index].depth;
				//gs_depth = max(gs_in[0].depth,max(gs_in[1].depth,gs_in[2].depth));
			}
			else{
				vec2 uv_max = gs_in[0].uv;
				float d_max = gs_in[0].pos_in.x;
				for (int idx = 1; idx < 3; ++idx){
					if (d_max < gs_in[idx].pos_in.x){
						d_max = gs_in[idx].pos_in.x;
						uv_max = gs_in[idx].uv;
					}
				}
				gs_uv = uv_max;
				gs_depth = d_max;
			}
			gl_Position = gl_in[index].gl_Position;
			EmitVertex();
		}

		void main() {
			float quality = get_quality();//mix(get_quality_depthdiff(),get_quality_15px(),0.5);
			for(int idx = 0; idx<3; idx++) // pixel inside the picture
				if(gl_in[idx].gl_Position.x < -1 || gl_in[idx].gl_Position.x > 1 || gl_in[idx].gl_Position.y < -1 || gl_in[idx].gl_Position.y > 1)
					return;
			for(int idx = 0; idx<3; idx++) // pixel before the image plane
				if(gl_in[idx].gl_Position.z < 0 )
					return;
			gen_vertex(0, quality);
			gen_vertex(1, quality);
			gen_vertex(2, quality);
			EndPrimitive();
		}
	)";
		}

		std::string ShadersList::getBlendingVertexShaderSource()
		{
			return R"(
		#version 420 core

		// two triangles to form a quad
		layout(location = 0) in vec2 position;

		out vec2 vs_position;

		void main(void)
		{
			vs_position = (1.0f+position.xy)/2.0f;
			gl_Position = vec4(position.x, position.y, 0.0f, 1.0f);
		}
	)";
		}

		std::string ShadersList::getBlendingFragmentShaderSource(int step)
		{
			return std::string(step == 1
				? R"(
			#version 420 core
			layout(location=3) out vec4 color;
			layout(location=4) out float quality;
		)"
				: R"(
			#version 420 core
			layout(location=5) out vec4 color;
			layout(location=6) out float quality;
		)") + R"(
		uniform sampler2D accumulator_image;
		uniform sampler2D new_image;

		uniform sampler2D accumulator_quality;

		uniform sampler2D new_triangle_quality;

		uniform sampler2D new_depth;

		uniform float blending_factor;
		uniform float input_distance;

		in vec2 vs_position;

		void main(void)
		{
			float accu_quality = texture(accumulator_quality, vs_position).x;

			float n_triangle_quality = texture(new_triangle_quality, vs_position).x;
			float n_depth = texture(new_depth, vs_position).x;

			vec4 accu_color = texture(accumulator_image, vs_position);
			vec4 n_color = texture(new_image, vs_position);

			float accu_w;
			float new_w;
			if(blending_factor > 0.5f)
			{
				accu_w = pow(accu_quality, blending_factor);
				new_w = pow(n_triangle_quality/n_depth/input_distance, blending_factor);
			}
			else{
				accu_w = 1.0;
				new_w = 1.0;
			}
			if (n_depth == 0.0f || n_triangle_quality == 0.0f)
				new_w=0.0f;
			if (accu_quality == 0.0f)
				accu_w=0.0f;

			vec4 tmp_color = accu_w*accu_color + new_w*n_color;
			if (accu_w + new_w >0.0f)
				tmp_color /= (accu_w + new_w);
			color = tmp_color;

			float tmp_quality = accu_w + new_w;
			if (tmp_quality > 0.0f)
			{
				if(blending_factor > 0.5f)
					{
						quality = pow(tmp_quality, 1.0f/blending_factor);
					}
					else{
						quality = tmp_quality;
					}
			}
			else 
				quality = 0.0f;
		}

	)";
		}
		std::string ShadersList::getSynthesisVertexShaderSourcePolynomial()
		{
			return R"(
		#version 420 core

		layout(location = 0) in float empty_value;
		uniform sampler2D depth_texture;
		uniform sampler2D mask_texture;
		uniform sampler2D polynomial1_texture;
		uniform sampler2D polynomial2_texture;
		uniform sampler2D polynomial3_texture;
		uniform sampler2D polynomial4_texture;
		uniform sampler2D polynomial5_texture;
		uniform sampler2D nl_output_mask;

		out VS_OUT {
			vec2 uv;
			float depth;
			vec3 pos_in; //for normal computation
			vec3 pos_out; //for normal computation
		} vs_out;

		uniform float w;
		uniform float h;
		uniform float n_w;
		uniform float n_h;
		uniform float sensor;


		// Rotation and translation
		uniform mat3 R;
		uniform vec3 t;

		// Input/output projection types 0 perspective, 1 erp, 2 Xslit
		uniform int erp_in;
		uniform int erp_out;

		// Perspective unprojection
		uniform vec2 f;
		uniform vec2 p;

		// Perspective projection
		uniform vec2 n_f;
		uniform vec2 n_p;


		uniform float max_depth;

		vec2 get_position_from_Vertex_ID(int id, float width) {
			int y = id / int(width);
			int x = id - y * int(width);
			return vec2(float(x), float(y));
		}

		vec2 get_pos_in_plane(){
			vec3 transl = -transpose(R)*t;
			vec2 uv = get_position_from_Vertex_ID(gl_VertexID, w);
			vec4 d = texture(depth_texture, vec2(uv.x/w, uv.y/h)).xyzw;
			mat2 m = mat2(d.x,0,0,d.x);//column major
			mat2 A = -transl.x/f.x*m+(1.f)*mat2(1.f,0.f,0.f,1.f);
			mat2 Ainv = inverse(A);
			return Ainv*(transl.yz+(transl.x/f.x)*(uv-p));
		}

		vec2 polynomial(vec2 st, vec4 p1, vec4 p2, vec4 p3, vec4 p4, vec4 p5){
		    float u =    st.x*st.x*st.x*p1.x
                        +st.x*st.x*st.y*p1.y
                        +st.x*st.y*st.y*p1.z
                        +st.y*st.y*st.y*p1.w
                        +st.x*st.x*p2.x
                        +st.x*st.y*p2.y
                        +st.y*st.y*p2.z
                        +st.x*p2.w
                        +st.y*p3.x;
		    float v =    st.x*st.x*st.x*p3.z
                        +st.x*st.x*st.y*p3.w
                        +st.x*st.y*st.y*p4.x
                        +st.y*st.y*st.y*p4.y
                        +st.x*st.x*p4.z
                        +st.x*st.y*p4.w
                        +st.y*st.y*p5.x
                        +st.x*p5.y
                        +st.y*p5.z;
            return vec2(u,v);
		}

		vec2 interpolate_polynomial(vec2 xy,vec2 st){
			//vec3 transl = -transpose(R)*t;
		    vec4 p1 = texture(polynomial1_texture, vec2(xy.x/w, xy.y/h)).xyzw;
		    vec4 p2 = texture(polynomial2_texture, vec2(xy.x/w, xy.y/h)).xyzw;
		    vec4 p3 = texture(polynomial3_texture, vec2(xy.x/w, xy.y/h)).xyzw;
		    vec4 p4 = texture(polynomial4_texture, vec2(xy.x/w, xy.y/h)).xyzw;
		    vec4 p5 = texture(polynomial5_texture, vec2(xy.x/w, xy.y/h)).xyzw;
		   // vec2 st = transl.yz;
		    return xy.xy+polynomial(st,p1,p2,p3,p4,p5) -p;
		}

		vec2 apply_matrix(vec2 t2){
			vec2 xy = get_position_from_Vertex_ID(gl_VertexID, w);
			float d = texture(depth_texture, vec2(xy.x/w, xy.y/h)).x;
			float mask = texture(mask_texture, vec2(xy.x/w, xy.y/h)).x;

			mat2 m = mat2(d,0,0.0,d);//column major
			vec2 uv = (m*t2)+xy/*+n_p*/-p;
			vs_out.uv = vec2(xy.x / w, xy.y / h);
			float depth =abs(f.x/m[0][0]);//0.5;

			vec3 transl = -transpose(R)*t;
		    vec2 st = transl.yz;
			if (mask < 0.1 ||
                    texture(mask_texture, vec2(xy.x/w, (xy.y-1)/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x-1)/w, xy.y/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x+1)/w, xy.y/h)).x < 0.1 ||
                    texture(mask_texture, vec2(xy.x/w, (xy.y+1)/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x-1)/w, (xy.y-1)/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x-1)/w, (xy.y+1)/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x+1)/w, (xy.y-1)/h)).x < 0.1 ||
                    texture(mask_texture, vec2((xy.x+1)/w, (xy.y+1)/h)).x < 0.1
                )
              {
                //return vec2(uv.x, uv.y);
                vs_out.depth = depth;
                return (m*t2)+xy-p;
              }
            else
              {
                vs_out.depth = depth;
                vec2 new_pos_uv = interpolate_polynomial(xy,t2);
				vec2 new_pos_xy = vec2((new_pos_uv.x+p.x)/w, (new_pos_uv.y+p.y)/h);
				float output_mask = texture(nl_output_mask, vec2(new_pos_xy.x, new_pos_xy.y)).x;
				if ( output_mask > 0.1)
					return new_pos_uv;
				else 
					return vec2(-w,-h);
              }
		}

		void apply_rotation(vec2 uv){


			vec3 dir = vec3(f.x,-uv.x,-uv.y);
			vec3 n_dir = /*transpose*/(R)*dir;
			float x = n_f.x/n_dir.x;
			n_dir = (x*n_dir)+vec3(0,n_w-n_p.x,n_h-n_p.y);
			gl_Position = vec4(-2.0*n_dir.y/n_w+1.0, 2.0*n_dir.z/n_h-1.0, 1., 1.);
		}

		void main(void)
		{
			vec2 t2 = get_pos_in_plane();
			vec2 uv =  apply_matrix(t2);
			apply_rotation(uv);
		}
	)";
		}
	}
}
