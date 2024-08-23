# RVS software

## Description

This software computes novel views.

## Authors

* Sarah Fachada, Universite Libre de Bruxelles, Bruxelles, Belgium
* Daniele Bonatto, Universite Libre de Bruxelles, Bruxelles, Belgium
* Arnaud Schenkel, Universite Libre de Bruxelles, Bruxelles, Belgium
* Bart Kroon, Koninklijke Philips N.V., Eindhoven, The Netherlands
* Bart Sonneveldt, Koninklijke Philips N.V., Eindhoven, The Netherlands

## License

```txt
The copyright in this software is being made available under the BSD
License, included below. This software may be subject to other third party
and contributor rights, including patent rights, and no such rights are
granted under this license.

Copyright (c) 2010-2018, ITU/ISO/IEC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
```

## Usage

### Commandline parameters

#### General parameters

| Cmd | Description |
|:----|:------------|
|     | json file path |
| --noopengl | using cpu |
| --analyzer |  analyse  |

#### Camera Json parameters

| Name         | Value         | Description |
|:-------------|:--------------|:------------|
|Name		   | string		   | camera name |
|Position      | float 3  	   | position (front,left,up) |
|Rotation      | float 3       | rotation (yaw,pitch,roll) |
|Depthmap      | int           | has a depth |
|Depth_range   | float 2       | min and max depth |
|Multi_depth_range| float 2    | min and max value in multidepth (non-Lambertian) (optional, default: Depth_range) |
|DisplacementMethod| string    | Depth or  Polynomial (optional, default: depth) |
|Resolution    | int 2    	   | resolution (pixel) |
|Projection    | string        | perspective or equirectangular |
|Focal         | float 2       | focal (pixel) |
|Principle_point| float 2      | principle point (pixel) |
|BitDepthColor | int           | color bit depth |
|BitDepthDepth | int           | depth map bit depth |
|ColorSpace    | string        | YUV420 or YUV400 |
|DepthColorSpace| string       | YUV420 or YUV400 |

#### View Synthesis Json parameters

| Name                     | Value       | Description |
|:-------------------------|:------------|:------------|
|Version                   | string      | version number |
|InputCameraParameterFile  | string      | filepath to input cameras json |
|VirtualCameraParameterFile| string      | filepath to input cameras json |
|VirtualPoseTraceName      | string      | filepath to posetraces (optional) |
|InputCameraNames          | string list | list of input cameras  |
|VirtualCameraNames        | string list | list of output cameras |
|ViewImageNames            | string list | filepaths to input images |
|DepthMapNames             | string list | filepaths to input depth |
|OutputFiles               | string list | filepaths to output images |
|StartFrame                | int         | first frame (starts at 0) |
|NumberOfFrames            | int         | number of frames in the input |
|NumberOfOutputFrames      | int         | number of frame in the output (optional, default: NumberOfFrames) |
|Precision                 | float       | precision level |
|ColorSpace                | string      | RGB or YUV working colorspace |
|ViewSynthesisMethod       | string      | Triangles |
|BlendingMethod            | string      | Simple or Multispectral |
|BlendingFactor            | float       | factor in the blending |

## References

* S. Fachada, D. Bonatto, A. Schenkel, G. Lafruit, View Synthesis with multiple reference views [M42343], San Diego, CA, US
* S. Fachada, D. Bonatto, A. Schenkel, G. Lafruit, Depth Image Based View Synthesis With Multiple Reference Views For Virtual Reality, 3DTV-Conference, Stockholm, Sweden.
* S. Fachada, B. Kroon, D. Bonatto, B. Sonneveldt, G. Lafruit, Reference View Synthesizer (RVS) 2.0 manual, [N17759], Ljubljana, Slovenia
* B. Kroon, Reference View Synthesizer (RVS) manual [N18068], ISO/IEC JTC1/SC29/WG11, Macau SAR, China, p. 19, Oct. 2018.
* S. Fachada, D. Bonatto, M. Teratani, G. Lafruit, Light field rendering for non-Lambertian objects, Electronic Imaging 2021.

## Changelog

### v4.0 [m57104]

* Possibility to use a non-Lambertian map instead of a classical depth map for scenes including non-Lamnertian objects (on GPU only). Non-Lambertian maps modelise the pixel displacement by a polynomial equation instead of a single-channel disparity. Multi_depth_range and DisplacementMethod are added to the configuration files to specify the use of this feature.
* Slight change in quality definition in Geometry Shader 
* Can read pose traces longer than input files
* Corrected a loading function with YCrCb instead of YUV

