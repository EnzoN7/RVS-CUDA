#ifndef GpuDecoder_HPP
#define GpuDecoder_HPP

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <iostream>
#include <vector>
#include "dtype_conversion.cuh"
#include <thread>
#include <atomic>
#include <condition_variable> 
#include <string> 

#include <cuda.h> 
#include "NvDecoder/NvDecoder.h"
#include "../Utils/FFmpegDemuxer.h"

#define OUTPUT_TEST
#define USE_BUFFERING_DECODE 

#ifdef USE_BUFFERING_DECODE
#define DEC_BUFFER_SIZE 5
#else
#define DEC_BUFFER_SIZE 1
#endif

//============ NOTE: ONLY  WORK FOR INTRA-ONLY and Low dely P (IPPP..) profiles ============
using namespace std;
class GpuDecoder {
public:

    //Uses to give an uniqe Id to each instance of GpuDecoder
    static int instance_count;

    GpuDecoder();
    ~GpuDecoder();
    bool open(const std::string& filepath, int buf_size, int gpuId = 0, int start_frame_index = 0, int num_decoded_frames = 0);
    void start_buffring_thread(int video_total_number);
    bool decode_frame(int buff_idx, int frame_idx);
    void close();

    int id;
    int width;
    int height;
    int pixel_bytes;
    int buffer_size;
    int y_bytes;
    int uv_bytes;
    int frame_bytes;

    cudaStream_t stream;
    void* devYUV = nullptr;

#ifdef OUTPUT_TEST
    std::ofstream* outframe_test;
#endif // OUTPUT_TEST

    //Mutlti Theading variable
#ifdef USE_BUFFERING_DECODE
    void producer(int video_total_frame_num);
    std::atomic<bool> stop_flag{ false };
    std::atomic<int> write_idx{ 0 };
    std::atomic<int> read_idx{ 0 };
    std::atomic<int>* status;// [BUFFER_SIZE] ; // 0 = empty,1=writing,2 = full 
    std::atomic<int>* frame_idx;// [BUFFER_SIZE] ;
private:
    std::thread* producer_thread = nullptr;
#endif


private:

    static string makeCudaContext(CUcontext* cuCtx, int gpuId, unsigned int flags);

    int m_current_frame_index = 0;
    int m_start_frame_index = 0;
    int m_end_frame_index = INT32_MAX;

    void* m_devUV_tmp = nullptr;

    NvDecoder* m_decoder = nullptr;
    FFmpegDemuxer* m_demuxer = nullptr;
    CUcontext m_cuContext = nullptr;

    int m_video_stream_index;
    AVStream* m_video_stream = nullptr;

};

#endif 

