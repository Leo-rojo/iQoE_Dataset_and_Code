## Chunks generation:
* Put ToS original video in the folder original_video_short and run encode_original_video_in_various_rugs_1.py to generate the encoded videos in the folder encoded_video
* Run divide_in_dash_chunks_2.py and divide_in_dash_chunks_2_orig.py to generate the chunks in the folder chunks and chunks_orig, respectively for the encoded and original videos
* Run Inited_chunks_3.py to generate the inited chunks in the folder inited_dashed_files and inited_dashed_files_orig, respectively for the encoded and original videos, this process is needed to make the chunks playable by players
* Run tomkv_4.py and tomkv_4_orig.py to convert the inited chunks in mkv format in the folder mkvfiles and mkvfiles_original, respectively for the encoded and original videos

At the end of the process we will have all the chunks for the different encoding rugs of the ladder in the folder mkvfiles and the chunks for the original video in mkvfiles_orig, 
a version of already processed chunks can be downloaded from: https://drive.google.com/file/d/1yffyTB3nyYXFlEZOc-O0krFs-sEemSI8/view?usp=sharing