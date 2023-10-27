from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
#insert path to your folder Figure 5

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

path_iQoE='double_users'
#create folder points if does not exist
if not os.path.exists(path_iQoE+'\\figures'):
    os.makedirs(path_iQoE+'\\figures')
colori=cm.get_cmap('tab10').colors


users_scores=[]
users_videos=[]

for upfold in os.listdir(path_iQoE):
    if upfold.split('_')[0]=='user':
        user_scores=[]
        user_videos=[]
        for fold in os.listdir(path_iQoE+'\\'+upfold):
            identifier=fold.split('_')[-2]
            oldnew=fold.split('_')[-1]
            user_folder = path_iQoE+'\\'+upfold+'/user_' + identifier + '_'+oldnew
            print(user_folder)

            ##train data
            # save dictonary idx_original-score
            a=[]
            a2=[]
            d_train = {}
            #exp_orig_train_t = []
            #scaled_exp_orig_train_t = []
            #take first 20 of train
            with open(user_folder + '/Scores_' + identifier + '.txt') as f:
                for line in f:
                    val = line.split()[-1]
                    nextline = next(f)
                    key = nextline.split()[-1]
                    d_train[int(key)] = val
                    if len(d_train)==10:
                        break
            a.append([int(i) for i in list(d_train.values())])
            a2.append([int(i) for i in list(d_train.keys())])

            ###baseline data
            b=[]
            b2=[]
            d_baselines={}
            #exp_orig_train_b = []
            #scaled_exp_orig_train_b = []
            with open(user_folder + '/Scores_baseline' + identifier + '.txt') as f:
                for line in f:
                    val = line.split()[-1]
                    nextline = next(f)
                    key = nextline.split()[-1]
                    d_baselines[int(key)] = val
            b.append([int(i) for i in list(d_baselines.values())])
            b2.append([int(i) for i in list(d_baselines.keys())])

            ##test data
            #save dictonary idx_original-score
            c=[]
            c2=[]
            d_test = {}
            exp_orig_test=[]
            scaled_exp_orig_test=[]
            with open(user_folder+'/Scores_test_'+identifier+'.txt') as f:
                for line in f:
                   val = line.split()[-1]
                   nextline=next(f)
                   key = nextline.split()[-1]
                   d_test[int(key)] = val
            c.append([int(i) for i in list(d_test.values())])
            c2.append([int(i) for i in list(d_test.keys())])

            user_scores.append(a[0] + b[0] + c[0])
            user_videos.append(a2[0] + b2[0] + c2[0])
        users_scores.append(user_scores)
        users_videos.append(user_videos)

#userAn,userAo,userBn,userBo
for user_letter,single_user_scores in enumerate(users_scores):
    names=[['user_A_new','user_A_old'],['user_B_new','user_B_old']]
    colors=[['r','b'],['r','b']]#,colori[2],colori[4],colori[0]]
    stile=[['-','--'],['-','--']]#,':','-.','--']
    fig = plt.figure(figsize=(30, 10), dpi=100)
    for i,j in enumerate(single_user_scores):
        plt.plot(j, label=names[user_letter][i], linestyle=stile[user_letter][i], linewidth='7', color=colors[user_letter][i],marker='o', markersize=10, markeredgecolor='black')
    print('---------------------------------')
    plt.title('User scores')
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)  # add space left
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
    plt.xticks([i for i in range(len(user_videos[0])) if i%10==0],[user_videos[0][i] for i in range(len(user_videos[0])) if i%10==0])
    plt.xlabel('Video index in the pool')
    plt.yticks([1]+[i for i in range(0, 101, 10)])
    plt.ylabel('User score')
    ax.set_ylim([1, 101])
    plt.legend()
    plt.savefig(path_iQoE+'\\figures' + '/' + 'user_'+['A','B'][user_letter]+'_scores' + '.png')
    plt.close()

collect_him_corr=[]
for letter,user_scores in enumerate(users_scores):
    print(['user_A','user_B'][letter])
    #calculate pearson correlation between same user's scores
    corr_p, _ = pearsonr(user_scores[0], user_scores[1])
    print('Pearsons correlation: %.3f' % corr_p)
    #calculate spearman correlation between user_scores
    corr_s, _ = spearmanr(user_scores[0], user_scores[1])
    print('Spearmans correlation: %.3f' % corr_s)
    #calculate kendall correlation between user_scores
    corr_k, _ = kendalltau(user_scores[0], user_scores[1])
    print('Kendalls correlation: %.3f' % corr_k)
    collect_him_corr.append([corr_p,corr_s,corr_k])

######################################################################

#barplot
pers=[]
spears=[]
for ab,individual_corrs in enumerate(reversed(collect_him_corr)):
    pers.append(individual_corrs[0])
    spears.append(individual_corrs[1])
    kends=[individual_corrs[2]]

fig = plt.figure(figsize=(20, 10), dpi=100)
plt.bar([1,2,3,3.8,4.7,6.5,7.4,8,9,10], [0,0,0,0,0,0,0,0,0,0], align='center', color=['r', 'r', 'r', 'r'],label='Pearson')
plt.bar([1,2,3,3.8,4.7,6.5,7.4,8,9,10], [0,0,0,0,0,0,0,0,0,0], align='center', color=['g', 'g', 'g', 'g'],label='Spearman')
plt.bar([1,2,3,3.8,4.7,6.5,7.4,8,9,10], [0,0,0,pers[0],spears[0],pers[1],spears[1],0,0,0], align='center', color=['g', 'r', 'g', 'r'])
plt.xticks([4.25,6.95], ['Z8','Z53'])
plt.xlabel("Rater ", fontdict=font_axes_titles)
# plt.xlabel("Synthetic models",fontdict=font_axes_titles)
plt.ylabel('Correlation', fontdict=font_axes_titles)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
#plt.axhline(y=0, color='black', linestyle='-')
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 1.05])
legend1 = ax.legend(loc='center',bbox_to_anchor=[0.50, 1.05],ncol=3, frameon=False,fontsize = 50,handletextpad=0.3, columnspacing=0.5)
# plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
#plt.legend(ncol=3)
#plt.show()
plt.savefig(path_iQoE+'\\figures' + '/' + 'Fig_5b.pdf', bbox_inches='tight')
print('---------------------------------')

plt.close()



