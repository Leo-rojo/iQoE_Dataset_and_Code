import os
import json
import numpy as np
import csv

#add path of this folder, here is the path for my pc
#os.chdir('insrt path to Synthetic_users_implementation\\p1203_and_LSTM_models')

folder_videos_chunked='encoded_video_chunked'

#for each file that terminate with .npy in chunk_features_qp
for file in os.listdir('./chunk_features_not_qp'):
    if file[-3:]=='npy':
        npfile = np.load('chunk_features_not_qp/'+file)
        #get the name of the video
        group_name=file[3:]
        group_name=group_name.replace('.npy','')
        #print(group_name)
        #calculate qp of each video and add to npy file
        all_segments_qp=[]
        for video in os.listdir(folder_videos_chunked+'/'+group_name):
            video_with_path=folder_videos_chunked+'/'+group_name+'/'+video
            frames_qp = []
            os.system('python -m ffmpeg_debug_qp_parser '+ video_with_path  + ' output_file.json -m -of json -p ffmpeg-debug-qp -f')
            f = open('output_file.json')
            list_of_frames = json.load(f)
            for frame in list_of_frames:
                frames_qp.append(float(frame['qpAvg']))
            segment_qp = np.mean(frames_qp)
            all_segments_qp.append(segment_qp)
        #put qp in npfile
        newnpy = [c + [all_segments_qp[i]] for i,c in enumerate(npfile.tolist())]
        #save new npy file
        np.save('chunk_features_qp/'+file[:-4]+'_qp',newnpy)
        print(group_name+' done')


##############create features for biLSTM
#crea un file for each video experience which is composed by 7 chunks
th_exp=np.load('./exp_th.npy')
bb_exp=np.load('./exp_bb.npy')
mpc_exp=np.load('./exp_mpc.npy')

#format
#[st,bit][st,bit][st,bit]...30
def map_bit_to_features(bit,chunk_pos):
    VIDEO_BIT_RATE = [0.235, 0.375, 0.560, 0.750, 1.050, 1.750, 2.350, 3, 4.3, 5.8, 8.1, 11.6, 16.8]
    for file in os.listdir('./chunk_features_qp'):
        if file[-3:]=='npy':
            filetosplit=file[:-8]
            bit_name=float(filetosplit.split('_')[-1].replace('k', ''))
            #print(bit_name)
            if bit*1e3==bit_name:
                file_feat=np.load('./chunk_features_qp/'+file)
                #1 if best otherwise 0
                if bit_name==16800:
                    isb=1
                else:
                    isb=0
                rep=VIDEO_BIT_RATE.index(bit)
                ff=file_feat[chunk_pos]
                file_feat_converted=list(map(float, ff))
                #from file feat take pos
                return file_feat_converted

#size	width	height	br	psnr	ssim	vmaf  qp order of feat in csv
#th

th_exp_ordered=[]
for exp_nr in th_exp:
    th_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        th_exp_ordered_temp.append([rebuff,feat[-1],feat[3],feat[1]*feat[2],24])  #SD in sec,QP integer,BR in kilobits/s,RS wxh,FR=24
    th_exp_ordered.append(th_exp_ordered_temp)

#bb
bb_exp_ordered=[]
for exp_nr in bb_exp:
    bb_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        bb_exp_ordered_temp.append([rebuff,feat[-1],feat[3],feat[1]*feat[2],24])   #put features as WIV features: representation_index	rebuffering_duration	video_bitrate	chunk_duration	chunk_size	qp	framerate	width	height	is_best	psnr	ssim	vmaf
    bb_exp_ordered.append(bb_exp_ordered_temp)

#mpc
mpc_exp_ordered=[]
for exp_nr in mpc_exp:
    mpc_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        mpc_exp_ordered_temp.append([rebuff,feat[-1],feat[3],feat[1]*feat[2],24])   #put features as WIV features: representation_index	rebuffering_duration	video_bitrate	chunk_duration	chunk_size	qp	framerate	width	height	is_best	psnr	ssim	vmaf
    mpc_exp_ordered.append(mpc_exp_ordered_temp)

synthetic_experiences=th_exp_ordered+bb_exp_ordered+mpc_exp_ordered
#np.save('experiences_with_features_LSTM',total_experiences)

###############################################
#########################take 1000 from all
import numpy as np
import random
from random import randint
import os

#PARAMS
nr_of_exp=1000
nr_c=7


#modifiy bitrate form bit/s to kbit/s like in W4
for i in range(len(synthetic_experiences)):
    for k in range(len(synthetic_experiences[0])):
        synthetic_experiences[i][k][2]=synthetic_experiences[i][k][2]/1000

#collect experiences in form of elaborated features
random.seed(42)

list_of_exp_lstm=[]
list_of_exp_for_models=[]
for i in range(nr_of_exp):
    random_trace=randint(0, len(synthetic_experiences)-1)
    random_chunk=randint(0, len(synthetic_experiences[0])-nr_c)
    #list of exp to be used for training
    ch=[]
    for c in range(nr_c):
        ch=ch + synthetic_experiences[random_trace][random_chunk+c]
    list_of_exp_lstm.append(ch)

#np.save('feat_iQoE_for_synth_exp_LSTM',list_of_exp_lstm)#SD in sec,QP integer,BR in kilobits/s,RS wxh,FR=24
###############################################
#exp_thousand=np.load('feat_iQoE_for_synth_exp_LSTM.npy')
for count,exp in enumerate(list_of_exp_lstm):
    with open("exp" + str(count) + ".csv", "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
        oneexp=[]
        for i in range(0, 35, 5):
            oneexp.append(exp[i:i + 5])
        writer.writerows(oneexp)

    with open('filelist.txt', 'a+') as f:
        f.write('exp' + str(count) + '.csv' + '\n')


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

#remove all csv files
for file in os.listdir('.'):
    if file[-3:]=='csv':
        os.remove(file)
#remove filelist.txt
os.remove('filelist.txt')
#remove output.txt
os.remove('output.txt')

