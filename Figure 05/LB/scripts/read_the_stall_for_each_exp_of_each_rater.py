import os
import numpy as np

#insert path to your folder Figure 05

#load feat_iqoe.npy
feat_iqoe = np.load('../output_data/feat_iqoe.npy')

#for each user calculate the total stall for each experience
tot_stall_per_user_120 = []
for each_user in range(len(feat_iqoe)):
    #for each experience
    tot_stall = []
    for each_exp in feat_iqoe[each_user]:
        print(len(each_exp))
        reb = []
        for i in range(1, (1 + 4 * 10 - 1), 10):
            reb.append(float(each_exp[i]))
        tot_stall.append(sum(reb))
    tot_stall_per_user_120.append((sum(tot_stall)+8*120)/60)

upper_bound=max(tot_stall_per_user_120)+8*120
upper_bound_in_min=upper_bound/60


#for each user calculate the total stall for each experience
tot_stall_per_user_50 = []
for each_user in range(len(feat_iqoe)):
    #for each experience
    tot_stall = []
    for each_exp in feat_iqoe[each_user][0:50]:
        print(len(each_exp))
        reb = []
        for i in range(1, (1 + 4 * 10 - 1), 10):
            reb.append(float(each_exp[i]))
        tot_stall.append(sum(reb))
    tot_stall_per_user_50.append((sum(tot_stall)+8*50)/60)

upper_bound_50=max(tot_stall_per_user_50)+8*50
upper_bound_in_min_50=upper_bound_50/60

np.save('../output_data/tot_viewtime_per_user_120.npy', tot_stall_per_user_120)
np.save('../output_data/tot_viewtime_per_user_50.npy', tot_stall_per_user_50)