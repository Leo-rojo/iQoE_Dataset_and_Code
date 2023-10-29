import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dir_path=('input_data/time_over')
res=[]
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
l=[]
for i in res:
    y=i.split('.')
    l.append(y[0].split('_')[2])

print(l)



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

mod_names = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
bymod_mean=np.load('input_data/space_ave.npy')
bymod_std=np.load('input_data/space_std.npy')

n_queries=250
fig = plt.figure(figsize=(20, 10),dpi=100)
c=0
error_freq=np.arange(1,250,10)
for i in bymod_mean:
    plt.plot(i,label=mod_names[c],color='r',linewidth='7')
    plt.fill_between(np.arange(len(i)), i - bymod_std[c], i + bymod_std[c],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    c+=1

plt.xlabel("Iteration number", fontdict=font_axes_titles)
plt.xticks([0, 50, 100, 150, 200, 250], ['1', '50', '100', '150', '200', '250'])
plt.ylabel('Memory, KB', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.yticks(range(0,151,50))
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 150])
plt.savefig('./space_all.pdf',bbox_inches='tight')

bymod_mean_time=[]
bymod_std_time=[]
foru = []
for mod in mod_names:
    for u in range(32):
        foru.append(np.load('input_data/time_over/time_overhead_'+str(u)+mod+'.npy'))
bymod_mean_time.append(np.mean(foru, axis=0))
bymod_std_time.append(np.std(foru, axis=0))
#np.save('time_ave',bymod_mean_time)
#np.save('time_std',bymod_std_time)
fig1 = plt.figure(figsize=(20, 10),dpi=100)
c=0
for i in bymod_mean_time:
    plt.plot(i,label=mod_names[c],color='r',linewidth='7')
    plt.fill_between(np.arange(len(i)), i - bymod_std_time[c], i + bymod_std_time[c], alpha=0.5, edgecolor='#CC4F1B',
                     facecolor='#FF9848')
    c+=1
plt.xlabel("Iteration number", fontdict=font_axes_titles)
plt.xticks([0, 50, 100, 150, 200, 250], ['1', '50', '100', '150', '200', '250'])
plt.ylabel('Time, s', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.yticks(np.arange(0,0.31,0.05))#,['0','0.2','0.4','0.6','0.8','1','1.2','1.4','1.6'])
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 0.3])
plt.savefig('./time_all.pdf',bbox_inches='tight')
