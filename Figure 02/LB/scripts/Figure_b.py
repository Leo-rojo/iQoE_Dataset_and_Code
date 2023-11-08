import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.font_manager as font_manager
import os

#insert path to your folder

#take mos scores
mosarray=np.load('../output_data/mos_scores_hdtv.npy')
mosarray=[float(i) for i in mosarray]

#takes user scores and order them based on hightest discrepancy
collect_all=[]
users_scores=np.load('../output_data/users_scores_hdtv.npy')
medians=[]
#sort users_score by median
for i in range(32):
    medians.append(np.median(users_scores[i]))
#calculate min and max of medians
totusmin=min(medians)
totusmax=max(medians)
sorted_users=np.argsort(medians)

right_order=[i+1 for i in sorted_users]

#sort raters scores
ordered_users=[]
for num in right_order:
    ordered_users.append(users_scores[num - 1].tolist())

#ECDF for 4users and average user
lab=['H1','H2','H31','H32','HA']
my_pal = ["b",'grey', 'm',"c","#4b8521"]
fig=plt.figure(figsize=(20, 10),dpi=100)
subsec=[ordered_users[0],ordered_users[1],ordered_users[-2],ordered_users[-1],mosarray]
style=['-','-.',':','--','-']
#plot ecdf
for i in range(5):
    ecdf = sm.distributions.ECDF(subsec[i])
    plt.step(ecdf.x, ecdf.y,label=lab[i],color=my_pal[i],linewidth=6.0,linestyle=style[i])
plt.margins(0,0)
plt.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
plt.ylabel('% of scores', fontsize=55)
plt.xticks(fontsize=55)
plt.yticks(fontsize=55)
plt.xlabel('Score', fontsize=55)
plt.xticks([1,20,40,60,80,100])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['0','20','40','60','80','100'])
# HA_patch = mpatches.Patch(color='#4b8521', label='HA')
# H1_patch = mpatches.Patch(color='b', label='H1')
# H32_patch = mpatches.Patch(color='c', label='H32')
# H2_patch = mpatches.Patch(color='grey', label='H2')
# H31_patch = mpatches.Patch(color='m', label='H31')
font = font_manager.FontProperties(size=50)
#handles=[H1_patch,H2_patch,H31_patch,H32_patch,HA_patch],
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=50, ncol=5, framealpha=0, columnspacing=0.5,handletextpad=0.4,prop=font)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('../Figure_b.pdf',bbox_inches='tight')