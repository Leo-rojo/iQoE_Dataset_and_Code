## Datasets
The folder contains the datasets used in the project. The datasets are:
* dataset_34: it contains the 34 users from the initial part of the recruitment
* dataset_120: it contains the original 34 users plus users recruited through Microworkers
* dataset_128: it contains additional 8 users recruited through Microworkers

Each row of the datasets represents a user, while the column represent the features for each of the 4 chunk that compose the experience. Last column 'score' is the score assigned to experience from the users.
The features are described with c_1, c_2, c_3, c_4, where c_1 is the first chunk, c_2 the second and so on. The features are:
- rep_index: index of the chunk representation
- rebuffering_duration: rebuffering time in seconds
- video_bitrate: bitrate of the chunk in kbps
- video_chunk_size: size of the chunk in bytes
- width: video width
- height: video height
- is_best: flag that indicate if the representation is the best
- psnr: psnr value of the chunk
- ssim: ssim value of the chunk
- vmaf: vmaf value of the chunk
