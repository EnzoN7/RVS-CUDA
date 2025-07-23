# RVS-CUDA-NVDEC

## Description

This is the extended version of «RVS-CUDA», which supports compressed input texture views. For four input views, it achieves the same speed as when using raw YUV input texture views. It uses NDVDEC in the NVIDIA Video Codec SDK for GPU-based frame decoding.

#### Similar to RVS-CUDA 
* Only works for images in YUV420 format.
* The blending factor is always greater than or equal to 1.
* The blending method is always set to «Simple».
* The input files have an «Equirectangular» projection type.
* The output files are YUV texture files.

## Supported Features for the Compressed Input Texture
- H.264 and H.265 input
- Low Delay P and Intra Only configurations
- 8-bit and 10-bit input support (10-bit available only for H.265)

## Usage

### How to build (Tested on Windows)
Built with CMake (file: ```CMakeLists.txt```).
#### Dependencies
* [OpenCV (tested on v4.9.0)](https://github.com/opencv/opencv)
* [fmt (tested on v10.0.0)](https://github.com/fmtlib/fmt)
* [Catch2 (tested on v3.5.2)](https://github.com/catchorg/Catch2)
* [CUDA (tested on v12.4)](https://developer.nvidia.com/cuda-12-4-0-download-archive)
* [FFmpeg (Windows shared build v7.1.1)](https://www.gyan.dev/ffmpeg/builds/)
* [NVIDIA VIDEO CODEC SDK (v13.0)](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

#### CUDA architectures and corresponding compute capabilities

Adjust the value of ```CMAKE_CUDA_ARCHITECTURES``` in the ```CMakeLists.txt``` file according to your NVIDIA GPU architecture.

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
RVS-CUDA-NVDEC/
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
### How to run RVS-CUDA

```bash
cd RVS-CUDA-DEC/Build/
./Release/RVS.exe "../Config/RVS-{sequence}.json" [--fp16 || --fp64]
```
* --fp16 : Start RVS-CUDA in half bits precision (16 bits).
* --fp64 : Start RVS-CUDA in double bits precision (64 bits).
* *default* : Start RVS-CUDA in single bits precision (32 bits).

### How to configure the input texture type 
Use "TextureVideoType": "enc" parameter in the app config files (see .json files in the 'Config/app/' directory)
* See the .json files in the 'Config/app/' directory.

## Author of RVS-CUDA

Enzo Di Maria, Double Master's Degree | Specialist in Accelerated Computing:
* École de Technologie Supérieure, Montréal, Canada[^1]
* ENSEEIHT, Toulouse, France[^2]

Hossein Pejman
* École de Technologie Supérieure, Montréal, Canada[^1]

[^1]: ÉTS, https://www.etsmtl.ca
[^2]: ENSEEIHT, https://www.enseeiht.fr/fr/index.html

## Authors of the initial RVS software

* Sarah Fachada, Universite Libre de Bruxelles, Bruxelles, Belgium
* Daniele Bonatto, Universite Libre de Bruxelles, Bruxelles, Belgium
* Arnaud Schenkel, Universite Libre de Bruxelles, Bruxelles, Belgium
* Bart Kroon, Koninklijke Philips N.V., Eindhoven, The Netherlands
* Bart Sonneveldt, Koninklijke Philips N.V., Eindhoven, The Netherlands
