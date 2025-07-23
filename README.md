# RVS-CUDA-NVDEC

## Performance logs

**Goal: > 30 fps**

* **08 Avr: 24.85 fps**

* 25 Mar: 21.5 fps

* 09 Mar: 18.5 fps

* 08 Mar: 16.5 fps

* 28 Feb: 14 fps

* 10 Jan: 7 fps

## Description

This is the extended version of «RVS-CUDA», which supports compressed input texture views. For four input views, it achieves the same speed as when using raw YUV input texture views. It uses NDVDEC in the NVIDIA Video Codec SDK for GPU-based frame decoding.


## Support Features
-  YUV and compressed input texture view
- compressed H.264 and H.265 input
- Low Delay P and Intra Only configurations
- 8-bit and 10-bit input support (10-bit available only for H.265)

## Table of contents

- [Usage](#usage)[^2]
  - [How to build RVS-CUDA](#how-to-build-rvs-cuda)
    - [Dependencies](#dependencies)
    - [CUDA architectures and corresponding compute capabilities](#cuda-architectures-and-corresponding-compute-capabilities)
    - [Suggested project organization](#suggested-project-organization)
  - [How to run RVS-CUDA](#how-to-run-rvs-cuda)
  - [How to configure RVS-CUDA](#how-to-configure-rvs-cuda)
    - [View synthesis *.json* parameters](#view-synthesis-json-parameters--rvs-sequencejson)
    - [Camera *.json* parameters](#camera-json-parameters--sequencejson)
    - [Differences from the initial version](#differences-from-the-initial-version)
- [Example of view synthesis using RVS on «ClassroomVideo»](#example-of-view-synthesis-using-rvs-on-classroomvideo)
- [Quality and performance](#quality-and-performance)
  - [Comparison of time for producing a «Perspective» type image](#comparison-of-time-for-producing-a-perspective-type-image)
  - [Quality of different implementations on «ClassroomVideo»](#quality-of-different-implementations-on-classroomvideo)
  - [Quality of different implementations on «Museum»](#quality-of-different-implementations-on-museum)
  - [Quality of different implementations on «Chess»](#quality-of-different-implementations-on-chess)
- [Author of RVS-CUDA](#author-of-rvs-cuda)
- [Authors of the initial RVS software](#authors-of-the-initial-rvs-software)

## Usage

### How to build RVS-CUDA (Tested on Windows)

#### Dependencies
Built with CMake (file: ```CMakeLists.txt```).
* [OpenCV (tested on v4.9.0)](https://github.com/opencv/opencv)
* [fmt (tested on v10.0.0)](https://github.com/fmtlib/fmt)
* [Catch2 (tested on v3.5.2)](https://github.com/catchorg/Catch2)
* [CUDA (tested on v12.4)](https://developer.nvidia.com/cuda-12-4-0-download-archive)
* [FFmpeg (ffmpeg windows shared build v7.1.1)](https://www.gyan.dev/ffmpeg/builds/)
* [NVIDIA VIDEO CODEC SDK (v13.0)](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

#### CUDA architectures and corresponding compute capabilities

Adjust the value of ```CMAKE_CUDA_ARCHITECTURES``` in the ```CMakeLists.txt``` file according to your NVIDIA GPU architecture[^9].

| Compute Capability | Architecture       | Example GPUs                          |
|--------------------|-------------------|---------------------------------------|
| **30**             | Kepler            | GTX 780, Tesla K20                    |
| **32**             | Kepler            | Tegra K1                              |
| **35**             | Kepler            | Tesla K40, GTX 770                    |
| **37**             | Kepler            | Tesla K80                             |
| **50**             | Maxwell           | GTX 750, GTX 750 Ti                   |
| **52**             | Maxwell           | GTX 970, GTX 980                      |
| **53**             | Maxwell           | Tegra X1                              |
| **60**             | Pascal            | GTX 1080, GTX 1070                    |
| **61**             | Pascal            | GTX 1050, GTX 1060                    |
| **62**             | Pascal            | Jetson TX2                            |
| **70**             | Volta             | Tesla V100                            |
| **72**             | Volta             | Xavier AGX, Jetson AGX                |
| **75**             | Turing            | RTX 2060, RTX 2070, RTX 2080, GTX 1660|
| **80**             | Ampere            | A100, RTX 3090, RTX 3080              |
| **86**             | Ampere            | RTX 3060, RTX 3070, RTX 3080 Ti       |
| **89**             | Ada Lovelace      | RTX 4090, RTX 4080                    |
| **90**             | Hopper            | H100, Hopper GPUs                     |

#### Suggested project organization

```
RVS-CUDA/
├── Build/
│   └── Release/
│       └── RVS.exe
├── Config/
│   ├── app  (*.json)
│   ├── camera (*.json)
│   └── pose_traces (*.csv)
├── sequence/
│   └── {A01 (*.yuv)}/
│       └─── {hevc (*.mp4)}/
│   └── {B01 (*.yuv)}/
│       └── {hevc (*.mp4)}/
│   └── {...  (*.yuv)}/
│       └── {hevc (*.mp4)}/
├── rvs/
│   └── src/
└── rvs_cuda_lib/
    └── src/
```

* ```sequence/``` folder = Input files

### How to run RVS-CUDA

```bash
cd RVS-CUDA/Build/
./Release/RVS.exe "../Config/RVS-{sequence}.json" [--fp16 || --fp64]
```
* --fp16 : Start RVS-CUDA in half bits precision (16 bits).
* --fp64 : Start RVS-CUDA in double bits precision (64 bits).
* *default* : Start RVS-CUDA in single bits precision (32 bits).

### How to configure RVS-CUDA

#### View synthesis *.json* parameters : ```RVS-{sequence}.json```

| Name                     | Value       | Description |
|:-------------------------|:------------|:------------|
|Version                   | string      | version number |
|InputCameraParameterFile  | string      | filepath to input cameras json |
|VirtualCameraParameterFile| string      | filepath to input cameras json |
|VirtualPoseTraceName      | string      | filepath to posetraces |
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

#### Camera *.json* parameters : ```{sequence}.json```

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

Enzo Di Maria, Double Master's Degree | Specialist in Accelerated Computing:
* École de Technologie Supérieure, Montréal, Canada[^7]
* ENSEEIHT, Toulouse, France[^8]

Hossein Pejman
* École de Technologie Supérieure, Montréal, Canada[^7]

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
