#include "cuda_runtime.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <iostream>

void convert_nv12_8bit_yuv420p( uint8_t* in_uv,  uint8_t* out_u, uint8_t* out_v, int width, int height, cudaStream_t& cudaStream);
void convert_nv12_10bit_yuv420p10le(uint8_t* in_y, uint8_t* in_uv, uint8_t* out_y, uint8_t* out_u, uint8_t* out_v, int width, int height, cudaStream_t& cudaStream);

