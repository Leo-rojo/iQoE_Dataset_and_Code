## Generate actual impaired video to be added in the iQoE web application:

#### input files:
* iQoE_synth_exp.npy 
* gif.gif for creating the waiting wheel

#### workflow:
1. Run 'Generate_all_videos.py' to generate 1000 videos experiences described by iQoE_synth_exp.npy, it needs a lot of time. The final videos will be saved in all_videos folder
2. Run 'rename_video_correctly.py' to rename the videos in order to make them compatible with the structure of iQoE web application
3. Run 'Generate_reference_videos.py' to generate 10 reference videos that will be saved in reference_videos folder

#### additional info
* In Generate_all_videos.py and Generate_reference_video.py you need to specify the path to the mkvfiles folder containing all the chunks for different encodings
* To generate the video files you need FFmpeg installed (our version is downloadable in https://www.gyan.dev/ffmpeg/builds)