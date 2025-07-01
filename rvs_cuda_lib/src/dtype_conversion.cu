

#include "dtype_conversion.cuh"

/*
__global__ void change_brightness(uint8_t* d_y, int width, int height, int pitch, int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t* row = d_y + y * pitch;
        row[x] = min(255, row[x] + brightness); // brighten pixel
    }
}

__global__ void invert_luma(uint8_t* y_plane, int width, int height, int pitch) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        y_plane[y * pitch + x] = 255 - y_plane[y * pitch + x];
    }
}*/


//=======================================================================================

template<class T>
__global__ void nv12_8bit_yuv420p_kernel(
    T* uv_in,
    T* u_out,
    T* v_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy UV plane (subsampled by 2)
    if (x < width / 2 && y < height / 2) {

        int offset_y_axis = y * (width / 2);

        u_out[offset_y_axis + x] = (T)(uv_in[2 * offset_y_axis + 2 * x]);     // U
        v_out[offset_y_axis + x] = (T)(uv_in[2 * offset_y_axis + 2 * x + 1]); // V
    }

}

void convert_nv12_8bit_yuv420p(uint8_t* in_uv, uint8_t* out_u, uint8_t* out_v, int width, int height, cudaStream_t& cudaStream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    nv12_8bit_yuv420p_kernel << <grid, block, 0, cudaStream >> > (in_uv, out_u, out_v, width, height);

    cudaStreamSynchronize(cudaStream);
}


//========================================================================


template<class T>
__global__ void nv12_10bit_to_yuv420p10le_kernel(
    T* y_in,
    T* uv_in,
    T* y_out,
    T* u_out,
    T* v_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //############## Each 16-bit little-endian sample has valid data in upper 10 bits ##############
    //We need to shift right by 6 bits to convert to 8-bit YUV42010P format

        // Copy Y plane (full resolution)
    if (x < width && y < height) {
        int idx = y * width + x;
        y_out[idx] = (T)(y_in[idx] >> 6);
    }
    
    // Copy UV plane (subsampled by 2)
    if (x < width / 2 && y < height / 2) {

        int offset_y_axis = y * (width / 2);

        u_out[offset_y_axis + x] = (T)(uv_in[2 * offset_y_axis + 2 * x] >> 6);     // U
        v_out[offset_y_axis + x] = (T)(uv_in[2 * offset_y_axis + 2 * x + 1] >> 6); // V

    }

}


//AV_PIX_FMT_P010LE to AV_PIX_FMT_YUV420P10LE
void convert_nv12_10bit_yuv420p10le(uint8_t* in_y, uint8_t* in_uv, 
    uint8_t* out_y, uint8_t* out_u, uint8_t* out_v, 
    int width, int height, cudaStream_t& cudaStream)
{

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // AVFrame* frame is in GPU memory (AV_PIX_FMT_CUDA / NV12)
    nv12_10bit_to_yuv420p10le_kernel << <grid, block,0, cudaStream >> > (
        (uint16_t*)in_y, (uint16_t*)in_uv,
        (uint16_t*)out_y, (uint16_t*)out_u, (uint16_t*)out_v,
        width, height);

    cudaStreamSynchronize(cudaStream);
 
}
