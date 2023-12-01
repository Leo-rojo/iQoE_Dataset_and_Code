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

#insert path to folder Figure 15

us_folder='../input_data/users'
res=[]
users_info = []
users_scores=[]
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        print(id)
        user_info=[]
        with open(us_folder+'/'+fold+'\\'+ 'save_personal_info.txt', 'r') as fp:
            #read each line of file
            for line in fp:
                l=line.replace('\n', '')
                if len(l.split('_'))>1:
                    if l.split('_')[1]=='':
                        user_info.append('None')
                    else:
                        user_info.append(l.split('_')[1])
                else:
                    user_info.append('None')
                print(l)
        current_user_info=[i for i in user_info]
        users_info.append(current_user_info)
        print('-----------------')

        user_folder=us_folder+'/'+fold
        d_test = {}
        exp_orig_test = []
        scaled_exp_orig_test = []
        with open(user_folder + '/Scores_test_' + id + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_test[int(key)] = val
        y_test = [int(i) for i in list(d_test.values())]

        ##train
        d_train = {}
        exp_orig_train = []
        scaled_exp_orig_train = []
        with open(user_folder + '/Scores_' + id + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_train[int(key)] = val
        y_train = [int(i) for i in list(d_train.values())]

        # baselines
        d_baselines = {}
        with open(user_folder + '/Scores_baseline' + id + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_baselines[int(key)] = val
        y_baseline = [int(i) for i in list(d_baselines.values())]

        users_scores.append(y_train+y_baseline+y_test)
        res.append(str(current_user_info[1])+'x'+str(current_user_info[2]))


#if elements in users_scores have the same value in res than make the average
#of the scores
res_df=pd.DataFrame(res)
users_scores_df=pd.DataFrame(users_scores)
users_scores_df['res']=res_df
users_scores_df=users_scores_df.groupby('res').mean()
users_scores_df=users_scores_df.reset_index()

#sort rows based on the resolution first parameter
users_scores_df['res1']=users_scores_df['res'].apply(lambda x: int(x.split('x')[0]))
#sort rows based on the resolution second parameter
users_scores_df['res2']=users_scores_df['res'].apply(lambda x: int(x.split('x')[1]))
#sort rows based on the resolution first parameter and second

users_scores_df=users_scores_df.sort_values(by=['res1','res2'])
print(users_scores_df)
#move the 11 row in the first position
#users_scores_df=users_scores_df.reindex([11,0,1,2,3,4,5,6,7,8,9,10,12])
#save column res in a list
res_list=users_scores_df['res'].tolist()
#remove first column
users_scores_df=users_scores_df.drop(users_scores_df.columns[0], axis=1)
#remove last 2 column
users_scores_df=users_scores_df.drop(users_scores_df.columns[-1], axis=1)
users_scores_df=users_scores_df.drop(users_scores_df.columns[-1], axis=1)
#put each row of data in a list
data_orig = []
#calculate nr of rows of users_scores_df
nr_rows=users_scores_df.shape[0]
for i in range(nr_rows):
    data_orig=data_orig+users_scores_df.iloc[i].tolist()

#repeat each element 120 times
res_list_extended=[i for i in res_list for j in range(120)]





#create datafram for boxplot
df = pd.DataFrame({'Resolution':res_list_extended,'ordered':data_orig})
df = df[['Resolution','ordered']]
dd=pd.melt(df,id_vars=['Resolution'],value_vars=['ordered'],var_name='')
my_colors = ["#ffd700"]

#plot boxplot
fig = plt.figure(figsize=(20, 10),dpi=100)
sns.set(rc={'figure.figsize':(8,3)},style="white")
sns.set_palette(my_colors)
ax=sns.boxplot(x='Resolution',y='value',data=dd,hue='')#,palette=my_pal
ax.tick_params(bottom=True)
ax.tick_params(left=True)
ax.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
#plt.locator_params(axis='x', nbins=5)
ax.set(ylabel='QoE')
#unique elements of res_list
print('resolutions_list')
for c,i in enumerate(res_list):
    print(c,i)



ax.set_xticklabels([str(i+1) for i in range(nr_rows)],fontsize=60,color='k')
ax.set_xticks([i for i in range(nr_rows) if i%4==0])
plt.yticks(fontsize=60, color='k')
plt.ylabel('Score', fontsize=60,color='k')
plt.xlabel('Resolution', fontsize=60, color='k')
plt.yticks([1, 20, 40, 60, 80, 100])
fig = ax.get_figure()
plt.margins(0,0)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)

#add small space between boxes of same group
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
adjust_box_widths(fig, 0.8)
ax.get_legend().remove()
#plt.savefig('resolutions_and_distributions.png',bbox_inches='tight')
plt.savefig('../resolutions_and_distributions.pdf',bbox_inches='tight')
plt.close()