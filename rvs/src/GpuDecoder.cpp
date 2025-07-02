#include "GpuDecoder.hpp"
#include <fstream>

int GpuDecoder::instance_count = 0;

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

GpuDecoder::GpuDecoder() {
    av_log_set_level(AV_LOG_ERROR);
}

GpuDecoder::~GpuDecoder() {

    close();
}

#ifdef USE_BUFFERING_DECODE

void GpuDecoder::producer(int video_total_frame_num)
{
    while (!stop_flag.load(std::memory_order_acquire))
    {
        int idx = write_idx.load(std::memory_order_relaxed) % buffer_size;

        // Wait until buffer is empty
        while (status[idx].load(std::memory_order_acquire) != 0)
        {
            if (stop_flag.load(std::memory_order_acquire)) return;
            std::this_thread::yield();
        }

        status[idx].store(1, std::memory_order_release); // mark as writing

        // Write data
        cudaStreamSynchronize(stream);
        if (decode_frame(idx, m_current_frame_index % video_total_frame_num))
        {
            frame_idx[idx].store(m_current_frame_index - 1, std::memory_order_release);

            // Write index advance should happen after frame is done
            write_idx.store((idx + 1) % buffer_size, std::memory_order_release);

            // Now make frame visible to consumer
            status[idx].store(2, std::memory_order_release); // mark as ready
        }
        else
        {
            frame_idx[idx].store(-1, std::memory_order_release);
            status[idx].store(0, std::memory_order_release); // mark as empty
        }
    }
}

/// <summary>
/// sart the thread to buffer the decoded frame
/// </summary>
/// <param name="video_total_frame_num">used for loop reading in multi threading mode</param>
void GpuDecoder::start_buffring_thread(int video_total_frame_num)
{
    stop_flag.store(false);
    write_idx.store(0);
    read_idx.store(0);
    status = new std::atomic<int>[buffer_size];
    frame_idx = new std::atomic<int>[buffer_size];
    for (int i = 0; i < buffer_size; i++)
    {
        status[i].store(0);
        frame_idx[i].store(-1);
    }

    cudaStreamSynchronize(stream);
    //decode_frame(0, 0);
    producer_thread = new std::thread(&GpuDecoder::producer, this, video_total_frame_num);


}

#endif

bool GpuDecoder::open(const std::string& filepath, int buf_size, int gpuId, int start_frame_index, int num_decoded_frames) {

    id = instance_count;
    instance_count++;

    //For the single thread decoding the size of buffer is always 1
    buffer_size = 1;

#ifdef USE_BUFFERING_DECODE
    buffer_size = buf_size;
#endif // !USE_BUFFERING_DECODE

    m_start_frame_index = start_frame_index;
    m_end_frame_index = num_decoded_frames > 0 ? (m_start_frame_index + num_decoded_frames) : INT32_MAX;
    m_current_frame_index = m_start_frame_index;


#ifdef OUTPUT_TEST
#ifdef USE_BUFFERING_DECODE
    outframe_test = new std::ofstream(std::string("..\\outputTest\\view_multi") + std::to_string(id) + ".yuv", std::ios::binary);
#else
    outframe_test = new std::ofstream(std::string("..\\outputTest\\view_single") + std::to_string(id) + ".yuv", std::ios::binary);
#endif // USE_BUFFERING_DECODE
#endif OUTPUT_TEST

    std::ifstream fpIn(filepath.c_str(), std::ios::in | std::ios::binary);
    if (fpIn.fail())
    {
        std::cerr << "Could not open input video file: " << filepath.c_str() << std::endl;
        exit(-1);
    }
    else
    {
        fpIn.close();
    }

    //Init Cuda driver APIs
    cuInit(0);

    int gpuCount = 0;
    cuDeviceGetCount(&gpuCount);
    if (gpuId < 0 || gpuId >= gpuCount)
    {
        std::cerr << "GPU Id is not in the valid range!" << std::endl;
        return false;
    }



    string deviceName = makeCudaContext(&m_cuContext, gpuId, 0);
    if (deviceName != "")
    {
        std::cout << "GPU " << deviceName << " is used for decoding." << std::endl;
    }
    else
    {
        std::cout << "Could not identify GPU device name.";
        return false;
    }

    //cuCtxPushCurrent(m_cuContext); //Not neccessary
    //Whether we push the current context or not this stream should be created after creating context to be valid for sending to the NvDecoder constructor
    //If we do not use this stream in the NVdecoder constructor using this stream would not be valid ...
    //... in the kenerl method like (nv12_10bit_to_yuv420p10le_kernel method) if we call decode_frame from another thread
    cudaStreamCreate(&stream);

    m_demuxer = new FFmpegDemuxer(filepath.c_str());
    m_video_stream_index = av_find_best_stream(m_demuxer->GetAVFormatContext(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (m_video_stream_index < 0) {
        std::cerr << "Cannot find video stream.\n";
        return false;
    }
    m_video_stream = m_demuxer->GetAVFormatContext()->streams[m_video_stream_index];

    bool low_latency = true;
    bool getDeviceFrame = true; //Get frame in the located in the GPU memory
    bool force_min_latency = true;
    cudaVideoCodec cuVCodec = FFmpeg2NvCodecId(m_demuxer->GetVideoCodec());
    m_decoder = new  NvDecoder(m_cuContext, getDeviceFrame, cuVCodec, low_latency, false, NULL, NULL, false, 0, 0, 1000, force_min_latency, 0, stream);

    pixel_bytes = m_demuxer->GetBitDepth() == 8 ? 1 : (m_demuxer->GetBitDepth() == 10 ? 2 : -1);
    AVPixelFormat pixel_fmt = m_demuxer->GetChromaFormat();

    if (!(pixel_fmt == AV_PIX_FMT_YUV420P || pixel_fmt == AV_PIX_FMT_YUV420P10LE) || !(pixel_bytes == 1 || pixel_bytes == 2))
    {
        std::cerr << "Pixel format not supported by the encoder.\n";
        return false;
    }

    width = m_demuxer->GetWidth();
    height = m_demuxer->GetHeight();
    y_bytes = width * height * pixel_bytes;
    uv_bytes = (width / 2) * (height / 2) * pixel_bytes;
    frame_bytes = y_bytes + 2 * uv_bytes;

    cudaStreamSynchronize(stream);
    cudaError_t cudaStatus = cudaMallocAsync(&devYUV, frame_bytes * buffer_size, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        return false;
    }

    cudaStatus = cudaMallocAsync(&m_devUV_tmp, 2 * uv_bytes, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    return true;
}

bool GpuDecoder::decode_frame(int buff_idx, int frame_idx) {

    if (frame_idx >= m_end_frame_index)
    {
        return false;
        std::cerr << "cannot read frame with index higher 'end_frame_index'.\n";
    }

    
    if (m_current_frame_index <= m_end_frame_index)
    {
        //Set position of reading frame (packet)
        //NOTE: This does not work if we move back (except the first frame) or skip some frames in the LD or RA videos
        if (m_current_frame_index != frame_idx)
        {
#ifdef _DEBUG
            //std::cout << "Skip decoding frame (view_id=" << id<<")[current_idx="<< m_current_frame_index<<", decoded_idx=" << frame_idx <<"]" << std::endl;
#endif
            m_current_frame_index = frame_idx;
            //AVRational frame_duration = av_inv_q(m_video_stream->avg_frame_rate)
            //OR
            AVRational frame_duration = { m_video_stream->avg_frame_rate.den, m_video_stream->avg_frame_rate.num };
            int64_t seek_pts = av_rescale_q(m_current_frame_index, frame_duration, m_video_stream->time_base);
            av_seek_frame(m_demuxer->GetAVFormatContext(), m_video_stream_index, seek_pts, AVSEEK_FLAG_BACKWARD);

            //Flush decoder
            m_decoder->Decode(nullptr, 0, CUVID_PKT_ENDOFSTREAM);
        }

        int64_t pt = 0;
        int decodedVideoBytes = 0;
        uint8_t* ptrVideo = nullptr;
        m_demuxer->Demux(&ptrVideo, &decodedVideoBytes, &pt);

        if (decodedVideoBytes > 0)
        {
            //CUVID_PKT_ENDOFPICTURE to signals the decoder that a complete packet has been sent to decode
            int numDecodedFrames = m_decoder->Decode(ptrVideo, decodedVideoBytes, CUVID_PKT_ENDOFPICTURE, m_current_frame_index);

            if (numDecodedFrames > 0)
            {
                if (decodedVideoBytes && numDecodedFrames != 1)
                {
                    std::cerr << "More than one frames were decoded once. Application only supports decoding Intra-only and IPPP videos" << std::endl;
                    exit(-1);
                }

                int framesizeInByte = m_decoder->GetFrameSize();
                int64_t FrameTimestamp = 0;
                uint8_t* decoded_frame = m_decoder->GetFrame(&FrameTimestamp);

                uint8_t* base = static_cast<uint8_t*>(devYUV) + frame_bytes * buff_idx;
                uint8_t* devY = base;
                uint8_t* devU = base + y_bytes;
                uint8_t* devV = base + (y_bytes + uv_bytes);

                cudaMemcpyAsync(
                    devY,
                    decoded_frame,
                    y_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream);

                cudaMemcpyAsync(
                    m_devUV_tmp,
                    (decoded_frame + y_bytes),
                    uv_bytes * 2,
                    cudaMemcpyDeviceToDevice,
                    stream);

                cudaStreamSynchronize(stream);

                if (pixel_bytes == 2)  //10-bit pixels
                {
                    convert_nv12_10bit_yuv420p10le(devY, (uint8_t*)m_devUV_tmp, devY, devU, devV, width, height, stream);
                }
                else if (pixel_bytes == 1) //8-bit pixels
                {
                    convert_nv12_8bit_yuv420p((uint8_t*)m_devUV_tmp, devU, devV, width, height, stream);
                }
#ifdef OUTPUT_TEST   

                // Write converted  YUV420P 8-bit or 10-bit (10LE) frame into the file
                // uint8_t* fb = new uint8_t[frame_bytes];
                // cudaMemcpy(fb, devYUV, frame_bytes, cudaMemcpyDeviceToHost);
                // outframe_test->write(reinterpret_cast<const char*>(fb), frame_bytes);
                
                //------------------------- Convert to the YUV420P 8-bit or 10-bit (10LE) ---------------
                /*uint8_t* y_b = new uint8_t[y_bytes];
                uint8_t* uv_b_tmp = new uint8_t[2*uv_bytes];
                uint8_t* uv_b = new uint8_t[2 * uv_bytes];
                cudaMemcpy(y_b, decoded_frame, y_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(uv_b_tmp, decoded_frame+ y_bytes, 2 * uv_bytes, cudaMemcpyDeviceToHost);

                uint16_t* yb_unit16 = (uint16_t*)y_b;
                for (int i = 0; i < y_bytes / 2; i++)
                    yb_unit16[i] = yb_unit16[i] >> 6;


                uint16_t* uv_b_temp_unit16 = (uint16_t*)uv_b_tmp;
                uint16_t* uv_b_unit16 = (uint16_t*)uv_b;
                for (int i = 0; i < uv_bytes/2; i++)
                {
                    uv_b_unit16[i] = uv_b_temp_unit16[2*i] >> 6;
                    uv_b_unit16[uv_bytes/2+i] = uv_b_temp_unit16[2*i+1]>> 6;
                }
                outframe_test->write(reinterpret_cast<const char*>(y_b), y_bytes);
                outframe_test->write(reinterpret_cast<const char*>(uv_b), uv_bytes*2);
                delete y_b;
                delete uv_b;
                delete uv_b_tmp;*/

                //When we call 
                //  outframe_test->write(reinterpret_cast<const char*>(outputframe->data[0]), y_bytes);
                // outframe_test->write(reinterpret_cast<const char*>(outputframe->data[1]), uv_bytes);
                //  outframe_test->write(reinterpret_cast<const char*>(outputframe->data[2]), uv_bytes);
#endif        

                m_current_frame_index++;
                return true;
            }
        }
    }
    return false;
}

void GpuDecoder::close() {

#ifdef USE_BUFFERING_DECODE
    this->stop_flag.store(true);
    if (this->producer_thread && this->producer_thread->joinable()) {
        this->producer_thread->join();
        delete this->producer_thread;
        this->producer_thread = nullptr;
    }
#endif // USE_BUFFERING_DECODE

  
    if (m_demuxer) delete m_demuxer;
    if (m_decoder) delete m_decoder;
    

    cudaFree(devYUV);
    cudaFree(m_devUV_tmp);
    cudaStreamDestroy(stream);

    //cuDevicePrimaryCtxRelease(gpuId);
    //m_decoder->~NvDecoder();
    //m_demuxer->~FFmpegDemuxer();

#ifdef OUTPUT_TEST
    outframe_test->close();
#endif
}


string GpuDecoder::makeCudaContext(CUcontext* cuCtx, int gpuId, unsigned int flags)
{
    CUresult result;
    CUdevice cuDevice = 0;
    char deviceName[255];

    result = cuDeviceGet(&cuDevice, gpuId);
    if (result != CUDA_SUCCESS)
    {
        return string("");
    }
    result = cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice);
    if (result != CUDA_SUCCESS)
    {
        return string("");
    }
    //This does not work for accessing gpu memomory if the context is called from the another thread.
    //result=cuCtxCreate(cuCtx, flags, cuDevice);
    result = cuDevicePrimaryCtxRetain(cuCtx, cuDevice);
    if (result != CUDA_SUCCESS)
    {
        return string("");
    }
    return string(deviceName);
}

