from itu_p1203 import P1203Standalone
import json
import numpy as np



exp_thousand=np.load('feat_iQoE_for_synth_exp.npy')
for count,exp in enumerate(exp_thousand):
    # create a dictionary with 3 keys called I11 I13 I23
    dict = {'I11': {'segments': []}, 'I13': {'segments': []}, 'I23': {'stalling': []}}

    reb=[]
    for i in range(1, 70, 10):
        reb.append(float(exp[i]))
    bit = []
    for i in range(2, 70, 10):
        bit.append(float(exp[i]))
    height = []
    for i in range(4, 70, 10):
        height.append(float(exp[i]))
    width = []
    for i in range(5, 70, 10):
        width.append(float(exp[i]))

    start = 0
    for i in range(7):
        seg_feat={"bitrate": bit[i],
        "codec": "h264",
        "duration": 4,
        "fps": 24.0,
        "resolution": str(int(width[i]))+'x'+str(int(height[i])),
        "start": start}

        dict['I13']['segments'].append(seg_feat)

        start+=4

    ts=[0,4,8,16,20,24]
    reb=reb[0:6]
    stallarray=[[ts[i],reb[i]] for i,x in enumerate(reb) if x!=0]
    if stallarray==[]:
        pass
    else:
        for eachstall in stallarray:
            dict['I23']['stalling'].append(eachstall)
    with open('exp'+str(count)+'.json', 'w') as fp:
        json.dump(dict, fp)

scoresp1203=[]
for conta in range(1000):
    print(conta)
    f = open('exp'+str(conta)+'.json')
    data = json.load(f)
    p1203_results = P1203Standalone(data).calculate_complete()
    onefive = p1203_results['O46']
    print(onefive)
    X_std = (onefive - 1) / (5 - 1)
    X_scaled = X_std * (100 - 1) + 1
    scoresp1203.append(X_scaled)

np.save('p1203_scores', scoresp1203)
#delete all files that start with "exp"
import os
for filename in os.listdir():
    if filename.startswith("exp"):
        os.remove(filename)







