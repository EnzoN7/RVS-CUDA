# RVS-CUDA

## Description

This new version of «RVS - Reference View Synthesizer»[^1] was developed as part of my final year project in Information Technology Engineering. By leveraging techniques based on asynchronism and kernel programming (via CUDA), this new iteration is approximately 100 times faster than RVS 4.0 on the «ClassroomVideo» sequence.

The main objective of this software is to generate synthesized views enabling 6DoF navigation within a virtual environment. The principle is as follows. Starting from real 360° footage captured by cameras arranged in an orderly manner, the challenge is to simulate, using interpolation techniques, any perspective of a virtual camera that can move freely within the entire space.

## Example of view synthesis using RVS on «ClassroomVideo»[^3]

**Order of reading: Ground Truth, OpenGL, OpenCV, CUDA**
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
  <img src="./Figures/A01-groundTruth.png" alt="Ground Truth" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-openGL.png" alt="OpenGL" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-openCV.png" alt="OpenCV (CPU)" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-CUDA.png" alt="CUDA" style="width: 45%; margin: 5px;">
</div>

## Quality and performance

### Comparison of time for producing a «Perspective» type image

| Sequence         | Input Views | CPU (ms) | OpenGL (ms) | CUDA (ms) | Speedup (CPU / CUDA) |
|------------------|-------------|----------|-------------|-----------|---------------------|
| ClassroomVideo  | 4           | 12870    | 1524        | **130.1** | 98.92               |
| Museum           | 11          | 28252    | 2926        | **509.0** | 55.50               |
| Chess            | 4           | 8034     | 1062        | **110.0** | 73.04               |

### Quality of different implementations on «ClassroomVideo»

| Implementation    | WS-PSNR[^4] (dB)                          | IV-PSNR[^5] (dB) | SSIM[^6]                           |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **CPU**   | Y: 33.61, U: 49.57, V: 52.21         | 44.16       | Y: 0.8287, U: 0.9913, V: 0.9947       |
| **OpenGL**| Y: 33.46, U: 49.12, V: 51.79         | 43.57       | Y: 0.8270, U: 0.9906, V: 0.9943       |
| **CUDA**  | Y: 33.43, U: 50.41, V: 52.79         | 44.51       | Y: 0.8243, U: 0.9929, V: 0.9954       |

### Quality of different implementations on «Museum»

| Implementation    | WS-PSNR (dB)                          | IV-PSNR (dB) | SSIM                                  |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **CPU**   | Y: 30.30, U: 38.84, V: 40.06         | 37.74       | Y: 0.9131, U: 0.9257, V: 0.9439       |
| **OpenGL**| Y: 29.42, U: 38.65, V: 39.87         | 36.36       | Y: 0.8989, U: 0.9250, V: 0.9434       |
| **CUDA**  | Y: 30.09, U: 38.04, V: 39.19         | 37.71       | Y: 0.9085, U: 0.9124, V: 0.9333       |

### Quality of different implementations on «Chess»

| Implementation    | WS-PSNR (dB)                          | IV-PSNR (dB) | SSIM                                  |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **CPU**   | Y: 23.58, U: 43.41, V: 46.91         | 32.04       | Y: 0.9251, U: 0.9889, V: 0.9947       |
| **OpenGL**| Y: 22.28, U: 41.56, V: 45.43         | 30.54       | Y: 0.9012, U: 0.9845, V: 0.9926       |
| **CUDA**  | Y: 23.28, U: 43.22, V: 46.98         | 31.76       | Y: 0.9240, U: 0.9882, V: 0.9944       |



## Usage[^2]

### Building and dependencies

Build with CMake (file: ```/rvs/CMakeLists.txt```).

* [OpenCV (tested on v4.9.0)](https://github.com/opencv/opencv)
* [fmt (tested on v10.0.0)](https://github.com/fmtlib/fmt)
* [glm (tested on v1.0.0)](https://github.com/g-truc/glm)
* [Catch2 (tested on v3.5.2)](https://github.com/catchorg/Catch2)

### CUDA architectures and corresponding compute capabilities[^9]

| Architecture       | Compute Capabilities |
|-------------------|---------------------|
| **Kepler**        | 30, 32, 35, 37      |
| **Maxwell**       | 50, 52, 53          |
| **Pascal**        | 60, 61, 62          |
| **Volta**         | 70, 72              |
| **Turing**        | 75                  |
| **Ampere**        | 80, 86              |
| **Ada Lovelace**  | 89                  |
| **Hopper**        | 90                  |

### Command line parameters

#### Example

```bash
/pathToBuildFolder/Release/RVS.exe /Config/RVS-A01.json [--cuda | --opengl]
```

#### General parameters

| Cmd | Description |
|:----|:------------|
|     | json file path |
|     | default version: CPU |
| --cuda | using CUDA speedup |
| --opengl | using OpenGL speedup |

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
|Precision                 | float       | precision level |
|ColorSpace                | string      | YUV working colorspace |
|ViewSynthesisMethod       | string      | Triangles |
|BlendingMethod            | string      | Simple |
|BlendingFactor            | float       | factor in the blending |

#### Camera Json parameters

| Name         | Value         | Description |
|:-------------|:--------------|:------------|
|Name		   | string		   | camera name |
|Position      | float 3  	   | position (front,left,up) |
|Rotation      | float 3       | rotation (yaw,pitch,roll) |
|Depthmap      | int           | has a depth |
|Depth_range   | float 2       | min and max depth |
|DisplacementMethod| string    | Depth |
|Resolution    | int 2    	   | resolution (pixel) |
|Projection    | string        | perspective or equirectangular |
|Focal         | float 2       | focal (pixel) |
|Principle_point| float 2      | principle point (pixel) |
|BitDepthColor | int           | color bit depth |
|BitDepthDepth | int           | depth map bit depth |
|ColorSpace    | string        | YUV420 |
|DepthColorSpace| string       | YUV420 |

#### Differences from the initial version

* Only works for images in YUV420 format.
* The blending factor is always greater than or equal to 1.
* The blending method is always set to «Simple».
* The input files have an «Equirectangular» projection type.
* The output files are YUV texture files.


## Author of RVS-CUDA

Enzo Di Maria, Double Master's Student:
* École de Technologie Supérieure, Montréal, Canada[^7]
* ENSEEIHT, Toulouse, France[^8]

[^1]: MPEG-I Visual, RVS, https://gitlab.com/mpeg-i-visual/rvs
[^2]: S. Fachada, B. Kroon, D. Bonatto, B. Sonneveldt, G. Lafruit, Reference View Synthesizer (RVS) 2.0 manual, [N17759], Ljubljana, Slovenia
[^3]: MPEG-I Visual, Content Database, https://mpeg-miv.org/index.php/content-database-2/
[^4]: Sun, Y., Lu, A., & Yu, L. (2017). Weighted-to-Spherically-Uniform Quality Evaluation for Omnidirectional Video. IEEE Signal Processing Letters, 24(9), 1-1. https://doi.org/10.1109/LSP.2017.2720693.
[^5]: Dziembowski, A., Mieloch, D., Stankowski, J., & Grzelka, A. (2022). IV-PSNR—The Objective Quality Metric for Immersive Video Applications. IEEE Transactions on Circuits and Systems for Video Technology, 32(11), 7575–7591. https://doi.org/10.1109/TCSVT.2022.3179575.
[^6]: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861.
[^7]: ÉTS, https://www.etsmtl.ca
[^8]: ENSEEIHT, https://www.enseeiht.fr/fr/index.html
[^9]: CUDA GPUs, Your GPU Compute Capability, https://developer.nvidia.com/cuda-gpus

## Authors of the initial RVS software

* Sarah Fachada, Universite Libre de Bruxelles, Bruxelles, Belgium
* Daniele Bonatto, Universite Libre de Bruxelles, Bruxelles, Belgium
* Arnaud Schenkel, Universite Libre de Bruxelles, Bruxelles, Belgium
* Bart Kroon, Koninklijke Philips N.V., Eindhoven, The Netherlands
* Bart Sonneveldt, Koninklijke Philips N.V., Eindhoven, The Netherlands

## License of the initial RVS software

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