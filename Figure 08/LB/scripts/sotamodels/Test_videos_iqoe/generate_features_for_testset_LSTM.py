import os
import json
import numpy as np
import csv

folder_videos_chunked= '' #insert the complete path to mkvfiles folder
ffmpeg_debug_qp_path=''#insert absolute path to ffmpeg-debug-qp folder

nr_chunks=4
nr_feat_ec=10
chunk_length=2
tot_nr_feat=nr_chunks*nr_feat_ec
video_nr=np.load('idx_col_test.npy')
feat_each_video=np.load('synth_exp_test.npy')
#take last column of feat_each_video
chunk_start_each_feat=feat_each_video[:,40]

#collect rep+poschunk for each of the 30 exp
all_reps_chnums=[]
for idx_exp,each_exp in enumerate(feat_each_video):
    initial_chunk=chunk_start_each_feat[idx_exp]
    reps=[]
    ch_nums=[]
    conta=1
    for i in range(0,nr_chunks*10,10):
        reps.append(each_exp[i])
        ch_nums.append("%.3d" % (initial_chunk+conta))
        conta+=1
    all_reps_chnums.append([reps,ch_nums])

#name of chunks to which we have to calculate the qp
chunks_of_test=[]
for i in all_reps_chnums:
    chunks_of_test.append([k for k in zip(i[0],i[1])])

#calculate qp for each chunk of each exp

all_video_qps=[]
for video in chunks_of_test:
    all_segments_qp=[]
    for chunk in video:
        name_chunk=str(int(chunk[0])+1)+'_'+str(chunk[1])
        video_with_path = folder_videos_chunked + '/' + name_chunk + '.mkv'
        print(video_with_path)
        frames_qp = []
        os.system('python -m ffmpeg_debug_qp_parser ' + video_with_path + ' output_file.json -m -of json -p '+ ffmpeg_debug_qp_path +' -f')
        f = open('output_file.json')
        list_of_frames = json.load(f)
        for frame in list_of_frames:
            frames_qp.append(float(frame['qpAvg']))
        segment_qp = np.mean(frames_qp)
        all_segments_qp.append(segment_qp)
    all_video_qps.append(all_segments_qp)

# extract from the 30 exp the features for lstm
feat_each_video_lstm=[]
exp_30 = feat_each_video
for count, exp in enumerate(exp_30):
    save_each_exp = []
    reb = []
    for i in range(1, tot_nr_feat, 10):
        reb.append(float(exp[i]))
    bit = []
    for i in range(2, tot_nr_feat, 10):
        bit.append(float(exp[i]))
    height = []
    for i in range(4, tot_nr_feat, 10):
        height.append(float(exp[i]))
    width = []
    for i in range(5, tot_nr_feat, 10):
        width.append(float(exp[i]))
    qps=all_video_qps[count]

    for i in range(0, nr_chunks):
        save_each_exp.append([reb[i], qps[i], bit[i], height[i]*width[i],24])
    feat_each_video_lstm.append(save_each_exp)


for count,exp in enumerate(feat_each_video_lstm):
    with open("exp" + str(video_nr[count]) + ".csv", "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
        for i in exp:
            writer.writerow(i)

    with open('filelist.txt', 'a+') as f:
        f.write('exp' + str(video_nr[count]) + '.csv' + '\n')

#calculate the biLSTM scores for each experience
filetxt='filelist.txt'
os.system('biQPS '+filetxt)
scoresbiqps=[]
with open('output.txt') as f:
    for line in f.readlines()[1:]:
        onefive=float(line.split('\t')[-1])
        X_std = (onefive - 1) / (5 - 1)
        X_scaled = X_std * (100 - 1) + 1
        scoresbiqps.append(X_scaled)
np.save('scoresbiqps',scoresbiqps)









