import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

#insert path to your folder

font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 60}
plt.rc('font', **font_general)

mae_all=np.load('input_data/save_mae_importance_for_all_hdtv.npy')
rmse_all=np.load('input_data/save_rmse_importance_for_all_hdtv.npy')

metric='mae'
legends = ['I', 'R', 'B', 'S', 'W', 'H', 'IS', 'P', 'SS', 'V']
legends = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


plt.figure(figsize=(20, 10),dpi=100)
#color bars with different colors in block of 7
#barplot group importances with group std
#mean of group importances
mean_importances=np.array(mae_all)[[0,7,25,28]].mean(axis=0)
std_importances=np.array(mae_all)[[0,7,25,28]].std(axis=0)
plt.bar(range(len(mean_importances)), mean_importances, color='r')#,yerr=std_importances)
plt.xticks(range(len(mean_importances)), legends)
plt.yticks(np.arange(0, 3.5, 0.5))
plt.ylabel('MAE increase', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
plt.xlabel('Shuffled influence factor', fontdict=font_axes_titles)
ax.set_ylim([0, 3])
plt.savefig('importance_hdtv_mae_mean_atypical.pdf', bbox_inches='tight')
plt.close()
#write mean_importances in csv for atypical users for each legend
df = pd.DataFrame(mean_importances, index=legends, columns=['mean_importances'])
df.to_csv('importance_hdtv_mae_mean_atypical.csv', sep=',', encoding='utf-8')

#rmse
metric='rmse'
plt.figure(figsize=(20, 10), dpi=100)
#color bars with different colors in block of 7
#barplot group importances with group std
#mean of group importances

plt.figure(figsize=(20, 10),dpi=100)
#color bars with different colors in block of 7
#barplot group importances with group std
#mean of group importances
mean_importances=np.array(rmse_all)[[0,7,25,28]].mean(axis=0)
std_importances=np.array(rmse_all)[[0,7,25,28]].std(axis=0)
plt.bar(range(len(mean_importances)), mean_importances, color='b')#,yerr=std_importances)
plt.xticks(range(len(mean_importances)), legends)
plt.yticks(np.arange(0, 3.5, 0.5))
plt.ylabel('RMSE increase', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
# plt.axhline(y=0, color='black', linestyle='-')
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
plt.xlabel('Shuffled influence factor', fontdict=font_axes_titles)
ax.set_ylim([0, 3])
plt.savefig('importance_hdtv_rmse_mean_atypical.pdf', bbox_inches='tight')
plt.close()

#write mean_importances in csv for atypical users for each legend
df = pd.DataFrame(mean_importances, index=legends, columns=['mean_importances'])
df.to_csv('importance_hdtv_rmse_mean_atypical.csv', sep=',', encoding='utf-8')
