REM ffprobe -v trace -show_packets -show_frames -show_format -show_streams output_intra.mp4
REM ffprobe -select_streams v:0 -show_packets -show_entries packet=pos, -of csv output_intra.mp4

mkdir hevc

set qp=27
set res=4096x2048
set type="INTRA"
set loop=0

REM FOR type="INTRA"
set param="qp="%qp%:keyint=1:min-keyint=1:no-scenecut=1"

REM FOR type="LDP"
REM set param="qp=%qp%:no-scenecut=1:bframes=0:ref=1"


set vi=v0
ffmpeg -f rawvideo -pix_fmt yuv420p10le -s:v %res% -r 30  -stream_loop %loop% -i %vi%_texture_%res%_yuv420p10le.yuv -c:v libx265 -x265-params "%param%" -preset medium -movflags +faststart -f mp4 hevc\%vi%_texture_%res%_yuv420p10le-qp%qp%-%type%.mp4

set vi=v1
ffmpeg  -f rawvideo -pix_fmt yuv420p10le -s:v %res% -r 30  -stream_loop %loop% -i %vi%_texture_%res%_yuv420p10le.yuv -c:v libx265 -x265-params "%param%" -preset medium -movflags +faststart -f mp4 hevc\%vi%_texture_%res%_yuv420p10le-qp%qp%-%type%.mp4

set vi=v2
ffmpeg   -f rawvideo -pix_fmt yuv420p10le -s:v %res% -r 30  -stream_loop %loop% -i %vi%_texture_%res%_yuv420p10le.yuv -c:v libx265 -x265-params "%param%" -preset medium -movflags +faststart -f mp4 hevc\%vi%_texture_%res%_yuv420p10le-qp%qp%-%type%.mp4

set vi=v5
ffmpeg   -f rawvideo -pix_fmt yuv420p10le -s:v %res% -r 30  -stream_loop %loop% -i %vi%_texture_%res%_yuv420p10le.yuv -c:v libx265 -x265-params "%param%" -preset medium -movflags +faststart -f mp4 hevc\%vi%_texture_%res%_yuv420p10le-qp%qp%-%type%.mp4

set vi=v6
ffmpeg  -f rawvideo -pix_fmt yuv420p10le -s:v %res% -r 30  -stream_loop %loop% -i %vi%_texture_%res%_yuv420p10le.yuv -c:v libx265 -x265-params "%param%" -preset medium -movflags +faststart -f mp4 hevc\%vi%_texture_%res%_yuv420p10le-qp%qp%-%type%.mp4


