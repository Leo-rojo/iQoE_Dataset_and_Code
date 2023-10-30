# Calculate features idx, vmaf, ssim, psnr, isbest, width, height, bitrate, stall, size
1. copy mkvfiles with generated videos in this folder. The folder is generated in Video_preparation folder. 
2. run 'create_folders_mkvfiles.py' to create the folders with the right structure 
3. put this new generated folders 1_,2_,3_,4_ in the folder mkvfiles
4. copy mkvfiles_original with generated videos in this folder. The folder is generated in Video_preparation folder.
5. run 'calculate_features_chunks_parall.py' to calculate the features for every chunks and save them in chunk_features folder

#### additional info
* file calculate_features_chunks_parall.py need the path to vmaf_4k_model in json format. Vmaf model can be downloaded from https://github.com/Netflix/vmaf/tree/master/model
* current mkvfiles folder is just an empty placeholder


