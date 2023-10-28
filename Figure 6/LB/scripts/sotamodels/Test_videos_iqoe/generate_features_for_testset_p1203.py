import numpy as np
from itu_p1203 import P1203Standalone
import json
import numpy as np
import os

nr_chunks=4
nr_feat_ec=10
chunk_length=2
tot_nr_feat=nr_chunks*nr_feat_ec
video_nr=np.load('idx_col_test.npy')
feat_each_video=np.load('synth_exp_test.npy')

#p1203 scores
exp_30=feat_each_video
for count,exp in enumerate(exp_30):
    # create a dictionary with 3 keys called I11 I13 I23
    dict = {'I11': {'segments': []}, 'I13': {'segments': []}, 'I23': {'stalling': []}}

    reb=[]
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

    start = 0
    for i in range(nr_chunks):
        seg_feat={"bitrate": bit[i],
        "codec": "h264",
        "duration": chunk_length,
        "fps": 24.0,
        "resolution": str(int(width[i]))+'x'+str(int(height[i])),
        "start": start}

        dict['I13']['segments'].append(seg_feat)

        start+=chunk_length

    ts=[0,2,4]#,8,20,24]
    reb=reb[0:3]
    stallarray=[[ts[i],reb[i]] for i,x in enumerate(reb) if x!=0]
    if stallarray==[]:
        pass
    else:
        for eachstall in stallarray:
            dict['I23']['stalling'].append(eachstall)
    with open('exp'+str(video_nr[count])+'.json', 'w') as fp:
        json.dump(dict, fp)

scoresp1203=[]
for conta in range(30):
    print(conta)
    f = open('exp'+str(video_nr[conta])+'.json')
    data = json.load(f)
    p1203_results = P1203Standalone(data).calculate_complete()
    onefive = p1203_results['O46']
    print(onefive)
    X_std = (onefive - 1) / (5 - 1)
    X_scaled = X_std * (100 - 1) + 1
    scoresp1203.append(X_scaled)

np.save('p1203_scores', scoresp1203)









