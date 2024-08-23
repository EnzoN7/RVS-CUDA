# RVS-CUDA

## Description

This new version of «RVS - Reference View Synthesizer»[^1] was developed as part of my final year project in Information Technology Engineering. By leveraging techniques based on asynchronism and kernel programming (via CUDA), this new iteration is approximately 100 times faster than RVS 4.0 on the «ClassroomVideo» sequence.

The main objective of this software is to generate synthesized views enabling 6DoF navigation within a virtual environment. The principle is as follows. Starting from real 360° footage captured by cameras arranged in an orderly manner, the challenge is to simulate, using interpolation techniques, any perspective of a virtual camera that can move freely within the entire space.

## Example of View Synthesis using RVS on «ClassroomVideo»

**Order of reading: Ground Truth, OpenGL, OpenCV, CUDA**
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
  <img src="./Figures/A01-groundTruth.png" alt="Ground Truth" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-openGL.png" alt="OpenGL" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-openCV.png" alt="OpenCV (CPU)" style="width: 45%; margin: 5px;">
  <img src="./Figures/A01-CUDA.png" alt="CUDA" style="width: 45%; margin: 5px;">
</div>

## Quality and performance

| Sequence         | Input Views | CPU (ms) | OpenGL (ms) | CUDA (ms) | Speedup (CPU / CUDA) |
|------------------|-------------|----------|-------------|-----------|---------------------|
| ClassroomVideo  | 4           | 12870    | 1524        | **130.1** | 98.92               |
| Museum           | 11          | 28252    | 2926        | **509.0** | 55.50               |
| Chess            | 4           | 8034     | 1062        | **110.0** | 73.04               |

### ClassroomVideo

| Implementation    | WS-PSNR (dB)                          | IV-PSNR (dB) | SSIM                                  |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **Version CPU**   | Y: 33.61, U: 49.57, V: 52.21         | 44.16       | Y: 0.8287, U: 0.9913, V: 0.9947       |
| **Version OpenGL**| Y: 33.46, U: 49.12, V: 51.79         | 43.57       | Y: 0.8270, U: 0.9906, V: 0.9943       |
| **Version CUDA**  | Y: 33.43, U: 50.41, V: 52.79         | 44.51       | Y: 0.8243, U: 0.9929, V: 0.9954       |

### Museum

| Implementation    | WS-PSNR (dB)                          | IV-PSNR (dB) | SSIM                                  |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **Version CPU**   | Y: 30.30, U: 38.84, V: 40.06         | 37.74       | Y: 0.9131, U: 0.9257, V: 0.9439       |
| **Version OpenGL**| Y: 29.42, U: 38.65, V: 39.87         | 36.36       | Y: 0.8989, U: 0.9250, V: 0.9434       |
| **Version CUDA**  | Y: 30.09, U: 38.04, V: 39.19         | 37.71       | Y: 0.9085, U: 0.9124, V: 0.9333       |

### Chess

| Implementation    | WS-PSNR (dB)                          | IV-PSNR (dB) | SSIM                                  |
|-------------------|--------------------------------------|-------------|---------------------------------------|
| **Version CPU**   | Y: 23.58, U: 43.41, V: 46.91         | 32.04       | Y: 0.9251, U: 0.9889, V: 0.9947       |
| **Version OpenGL**| Y: 22.28, U: 41.56, V: 45.43         | 30.54       | Y: 0.9012, U: 0.9845, V: 0.9926       |
| **Version CUDA**  | Y: 23.28, U: 43.22, V: 46.98         | 31.76       | Y: 0.9240, U: 0.9882, V: 0.9944       |



## Usage [^2]

### Commandline parameters

#### General parameters

| Cmd | Description |
|:----|:------------|
|     | json file path |
| --cuda | using CUDA speedup |
| --opengl | using OpenGL speedup |
|  | using only CPU |

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

## Author

* Enzo Di Maria, École de Technologie Supérieure, Montréal, Canada

## References

[^1]: MPEG-I Visual, RVS, https://gitlab.com/mpeg-i-visual/rvs
[^2]: S. Fachada, B. Kroon, D. Bonatto, B. Sonneveldt, G. Lafruit, Reference View Synthesizer (RVS) 2.0 manual, [N17759], Ljubljana, Slovenia