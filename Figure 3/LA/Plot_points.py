import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from k_means_constrained import KMeansConstrained
warnings.filterwarnings("ignore")
left, bottom, width, height = (0.88, -0.02, 0.13, 0.8)
colori=cm.get_cmap('tab10').colors
font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 70,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 70,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 70}
font_legend={'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)


#def run(k):
k=14
folder_points = './'
if not os.path.exists(folder_points):
    os.makedirs(folder_points)

def extract_colors(y_vect,spaced4):
    grad_col = []
    grad_mark = []
    for i in y_vect:
        if i <= spaced4[1]:
            grad_col.append('blue')
            grad_mark.append('o')
        elif i <= spaced4[2] and i > spaced4[1]:
            grad_col.append('cyan')
            grad_mark.append('v')
        elif i <= spaced4[3] and i > spaced4[2]:
            grad_col.append('green')
            grad_mark.append('d')
        elif i <= spaced4[4] and i > spaced4[3]:
            grad_col.append('yellow')
            grad_mark.append('p')
        elif i <= spaced4[5] and i > spaced4[4]:
            grad_col.append('red')
            grad_mark.append('s')
    return grad_col, grad_mark

#user 4
folders_queries='output_data'
queries_gs=np.load(folders_queries+'/save_queries_gs_4.npy')
queries_igs=np.load(folders_queries+'/save_queries_igs_4.npy')
queries_random=np.load(folders_queries+'/save_queries_random_4.npy')
queries_qbc=np.load(folders_queries+'/save_queries_qbc_4.npy')
queries_uc=np.load(folders_queries+'/save_queries_uc_4.npy')
all_points=np.load(folders_queries+'/all_space_points_user_4.npy').transpose()
y_train=np.load(folders_queries+'/y_train_user_4.npy')
y_igs=np.load(folders_queries+'/save_y_igs_4.npy')
y_gs=np.load(folders_queries+'/save_y_gs_4.npy')
y_random=np.load(folders_queries+'/save_y_random_4.npy')

# Input data
x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
total_area = (x_max - x_min) * (y_max - y_min)
#estimate pdf of all points
# Generate some data
np.random.seed(0)

kmeanModel = KMeansConstrained(n_clusters=k, size_min=2, n_init=50,random_state=26)  # 14,#19
kmeanModel.fit(all_points)
cluster_labels = kmeanModel.labels_

# print all_points+critical
fig = plt.figure(figsize=(20, 10), dpi=100)
x = all_points[:, 0]
y = all_points[:, 1]

#1-20, 21-40, 41-60, 61-80, and81-100
spaced4 = [1,20,40,60,80,100]#np.linspace(min(y_train), max(y_train), 5)
print('spaced4', spaced4)
grad_col = []
grad_mark = []
for i in y_train:
    if i <= spaced4[1]:
        grad_col.append('blue')
        grad_mark.append('o')
    elif i <= spaced4[2] and i > spaced4[1]:
        grad_col.append('cyan')
        grad_mark.append('v')
    elif i <= spaced4[3] and i > spaced4[2]:
        grad_col.append('green')
        grad_mark.append('d')
    elif i <= spaced4[4] and i > spaced4[3]:
        grad_col.append('yellow')
        grad_mark.append('p')
    elif i <= spaced4[5] and i > spaced4[4]:
        grad_col.append('red')
        grad_mark.append('s')

plt.margins(0.02, 0.01)
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
plt.ylabel('TotalStall', fontdict=font_axes_titles)
plt.xlabel('TotalVMAF', fontdict=font_axes_titles)
ax.set_ylim([-0.05, 1.05])

for xp, yp, m, c in zip(x, y, grad_mark, grad_col):
    # if xp in ep_x and yp in ep_y:
    #     scatter = ax.scatter(xp, yp, linestyle='None', s=300, marker=m, c=c, edgecolors='blue', linewidth=4)
    # else:
    scatter = ax.scatter(xp, yp, linestyle='None', s=1200, marker=m, c=c, edgecolors='black')  # cmap=cmap


plt.savefig(folder_points + '/trajectory_all.pdf', bbox_inches='tight')
plt.close()

#################################################################

names=['igs','uc','gs','qbc','random']
aggregate_results=[]
for count,ss in enumerate([queries_igs,queries_uc,queries_gs,queries_qbc,queries_random]):

    i=ss[0]
    if names[count]=='random' or names[count]=='igs' or names[count]=='gs':
        for nr_qconsidered in [50]:#5,10,15,20,25,30,35,
            fig = plt.figure(figsize=(20, 10), dpi=100)
            traj = i[:nr_qconsidered]
            if names[count]=='random':
                gc,gm=extract_colors(y_random[:nr_qconsidered],spaced4)
            elif names[count]=='igs':
                gc,gm=extract_colors(y_igs[:nr_qconsidered],spaced4)
            elif names[count]=='gs':
                gc,gm=extract_colors(y_gs[:nr_qconsidered],spaced4)
            #color = [i for i in range(len(traj))]
            x_ss=[i[0] for i in traj]
            y_ss=[i[1] for i in traj]

            plt.margins(0.02, 0.01)
            ax = plt.gca()
            ax.tick_params(axis='x', which='major', width=7, length=24)
            ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
            plt.ylabel('TotalStall', fontdict=font_axes_titles)
            plt.xlabel('TotalVMAF', fontdict=font_axes_titles)
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-0.01, 1.01])

            for xp, yp, m, c in zip(x_ss, y_ss, gm, gc):
                # if xp in ep_x and yp in ep_y:
                #     scatter = ax.scatter(xp, yp, linestyle='None', s=300, marker=m, c=c, edgecolors='blue', linewidth=4)
                # else:
                scatter = ax.scatter(xp, yp, linestyle='None', s=1200, marker=m, c=c, edgecolors='black')  # cmap=cmap

            plt.savefig(folder_points + '/trajectory' + names[count]+'_'+str(nr_qconsidered)+'.pdf', bbox_inches='tight')
            plt.close()

print('figures done')