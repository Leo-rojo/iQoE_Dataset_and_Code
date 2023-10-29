import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
from matplotlib.patches import PathPatch
import statsmodels.api as sm
from matplotlib import cm
import time

us_folder='../input_data/raters'

#delete files called save_time_nonone and save_time_correct
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        for i in ['save_time_nonone.txt','save_time_correct.txt','save_time_correct2.txt']:
            try:
                os.remove(us_folder+'/'+fold+'\\'+ i )
            except OSError:
                pass

#remove lines that have value none after'_' from file
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        with open(us_folder+'/'+fold+'\\'+ 'save_time.txt', 'r') as f:
            lines = f.readlines()
            lines = [l for l in lines if l.split('_')[-1].replace('\n', '') != 'None']
            with open(us_folder+'/'+fold+'\\'+"save_time_nonone.txt", "w") as f1:
                f1.writelines(lines)

#unfortunately, some raters just paused the experience without pressing the button pause (or they just closed the window or they let it open for a long time without interacting)
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        print(id)
        user_info=[]
        with open(us_folder+'/'+fold+'\\'+ 'save_time_nonone.txt', 'r') as fp:
            linesfp = [line.rstrip() for line in fp]
            #read each line of file
            for line_nr in range(len(linesfp)):
                line=linesfp[line_nr]
                val=line.split('_')[-1]
                if val[0]=='e':
                    date_time = time.gmtime(float(val.replace('experience', '')))
                    with open(us_folder + '/' + fold + '\\' + "save_time_correct.txt", "a") as f1:
                        f1.writelines(line + '\n')
                elif line.split('_')[-2]=='pause':
                    with open(us_folder + '/' + fold + '\\' + "save_time_correct.txt", "a") as f1:
                        f1.writelines(line + '\n')
                    break
                else:
                    date_time = time.gmtime(float(val))

                    #read next line after line

                    line2 = linesfp[line_nr+1]
                    #print(line2)
                    val2=line2.split('_')[-1]
                    if val2[0]=='e':
                        date_time2 = time.gmtime(float(val2.replace('experience', '')))
                    else:
                        date_time2 = time.gmtime(float(val2))
                    #calculate the difference between the two times in seconds!!!
                    diff = time.mktime(date_time2) - time.mktime(date_time)
                    #diff in secs
                    with open(us_folder+'/'+fold+'\\'+ "save_time_correct.txt", "a") as f1:
                        if diff>60*5: #five mins of inactivity
                            #add current line+pause with this + restart with next line
                            f1.writelines(line+'\n')
                            if val[0]=='e':
                                val = val.replace('experience', '')
                            if val2[0]=='e':
                                val2 = val2.replace('experience', '')
                            f1.writelines('pause_'+val+'\n')
                            f1.writelines('restart_'+val2+'\n')
                        else:
                            f1.writelines(line+'\n')

        #check for restart alone
        with open(us_folder+'/'+fold+'\\'+ 'save_time_correct.txt', 'r') as fp:
            #read each line of file
            for line in fp:
                nameref=line.split('_')[-2]
                value=line.split('_')[-1]

        # Define the file paths
        input_file = "save_time_correct.txt"
        output_file = "save_time_correct2.txt"

        # Open the input file for reading and the output file for writing
        with open(us_folder+'/'+fold+'\\'+ input_file, 'r') as f_in, open(us_folder+'/'+fold+'\\'+ output_file, 'w') as f_out:
            last_line_started_with_pause = False
            val=0
            for line in f_in:
                if line.split('_')[-2]=='pause':
                    last_line_started_with_pause = True
                elif line.split('_')[-2]=='restart':
                    if not last_line_started_with_pause:
                        f_out.write(('pause_'+val))
                    last_line_started_with_pause = False
                val = line.split('_')[-1]
                f_out.write(line)

#





#
res=[]
ids=[]
users_info = []
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        ids.append(id)
        print(id)
        user_info=[]
        with open(us_folder+'/'+fold+'\\'+ 'save_time_correct2.txt', 'r') as fp:
            #read each line of file
            for line in fp:
                nameref=line.split('_')[-2]
                value=line.split('_')[-1]
                if nameref=='start':
                    user_info.append(value.replace('\n', ''))
                if nameref=='pause' and value[0]=='e':
                    user_info.append(value.replace('experience', '').replace('\n', ''))
                if nameref=='restart':
                    user_info.append(value.replace('\n', ''))
                if nameref=='pause' and value[0]!='e':
                    user_info.append(value.replace('\n', ''))
        users_info.append(user_info)
        print('-----------------')
users=[]
for i in range(len(users_info)):
    user=[]
    for k in range(len(users_info[i])):
        date_time = time.gmtime(float(users_info[i][k]))
        dts=time.mktime(date_time)
        user.append(dts)
    users.append(user)

#calculate the differences
users_diff=[]
for i in users:
    dif_us=[]
    for k in range(1,len(i),2):
        dif_us.append(i[k]-i[k-1])
    users_diff.append(dif_us)

#array sum of differences
final=[sum(i) for i in users_diff]
#convert seconds in minutes
final=[i/60 for i in final]
print([i for i in zip(ids, final)])

#plot ecdf
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
from matplotlib import cm
colori=cm.get_cmap('tab10').colors

vt_50=np.load('../output_data/tot_viewtime_per_user_50.npy', allow_pickle=True)
vt_120=np.load('../output_data/tot_viewtime_per_user_120.npy', allow_pickle=True)

fig,ax = plt.subplots(figsize=(20, 10), dpi=100)

ecdf_b = sm.distributions.ECDF(vt_50)
plt.step(ecdf_b.x, ecdf_b.y, label='Playback for iQoE', linewidth=7.0,color='b', linestyle='--')#, linestyle=stile[conta]
ecdf_g = sm.distributions.ECDF(vt_120)
plt.step(ecdf_g.x, ecdf_g.y, label='Total playback', linewidth=7.0,color='g', linestyle='-.')#, linestyle=stile[conta]
ax.hlines(y=0, xmin=0, xmax=min(final)+0.4, linewidth=7, color='r')
ax.hlines(y=0, xmin=0, xmax=min(vt_120)+0.4, linewidth=7, color='g')
ax.hlines(y=0, xmin=0, xmax=min(vt_50)+0.4, linewidth=7, color='b')
ax.hlines(y=1, xmin=max(vt_50), xmax=113.3, linewidth=7, color='b')
ax.hlines(y=1, xmin=max(vt_120), xmax=188, linewidth=7, color='g')
ecdf_r = sm.distributions.ECDF(final)
plt.step(ecdf_r.x, ecdf_r.y, label='Series completion', linewidth=7.0,color='r', linestyle='-')#, linestyle=stile[conta]




ax.vlines(x=min(final), ymin=0, ymax=ecdf_r.y[1], linewidth=7, color='r')
ax.vlines(x=min(vt_50), ymin=0, ymax=ecdf_b.y[1], linewidth=7, color='b')
ax.vlines(x=min(vt_120), ymin=0, ymax=ecdf_g.y[1], linewidth=7, color='g')
#ax.hlines(y=1, xmin=max(final), xmax=120, linewidth=7, color='r')

plt.xlabel('Time, minutes', fontdict=font_axes_titles)
plt.ylabel('% of raters', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
plt.xticks([i for i in range(0,190,30)])
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
#add xlim
ax.set_xlim([0, 190])
# if metric=='mae':
#     ax.set_xlim([0, 50])
# else:
#     ax.set_xlim([0, 60])
# # plt.xlim(0, 25)
# plt.show()
plt.legend(loc='lower right', prop={'size': 50},frameon=False)
plt.savefig('../Fig_5c.pdf', bbox_inches='tight')
plt.close()

#median of final
print(np.median(final))
#min of final
print(min(final))
#max of final
print(max(final))
#min of vt_120
print(min(vt_120))
#max of vt_120
print(max(vt_120))
#min of vt_50
print(min(vt_50))
#max of vt_50
print(max(vt_50))

sorted_by_final = sorted(zip(ids, final), key=lambda x: x[1])
print(sorted_by_final)

# Create a list of tuples where each tuple contains the original index and the corresponding (id, final) pair
indexed_list = list(enumerate(zip(ids, final)))
# Sort the list of tuples based on the 'final' value
sorted_by_final = sorted(indexed_list, key=lambda x: x[1][1])
# Extract the indices from the sorted list
indices = [item[0] for item in sorted_by_final]
#sort user info base on indices
# users_info_sorted=[users_info[i] for i in indices]
# for i in users_info_sorted:
#     print(len(i))

#sort users_info




