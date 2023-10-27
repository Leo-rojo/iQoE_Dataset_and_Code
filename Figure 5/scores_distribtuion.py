import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
from matplotlib.patches import PathPatch
import statsmodels.api as sm

#insert path to your folder Figure 5
isc=[]
identifiers=[]
#collect individual scores and mos
for fold in os.listdir('dataset'):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        print(identifier)
        identifiers.append(identifier)
        #collect scores
        xls = pd.ExcelFile(r"dataset/user_"+identifier) #use r before absolute file path
        sheetX = xls.parse(0)
        iarray=sheetX['score'].tolist()
        isc.append(iarray)

mosarray=np.mean(isc,axis=0)
collect_all=[]
users_scores=np.array(isc).reshape(len(isc),120)

#ave each user
aveeachuserarray=np.median(isc,axis=1)
#sort index of aveeachuserarray
index=np.argsort(aveeachuserarray)
#repeated users position
sort_median_users=np.array(identifiers)[index]
#print identifiers and theri median
for i in range(len(sort_median_users)):
    print(sort_median_users[i],aveeachuserarray[index][i])

#collect data for description
sorted_for_data=aveeachuserarray[index]
print('atypical median')
for i in range(len(sorted_for_data)):
    if i<6 or i>120-7:
        print('z'+str(i+1),sorted_for_data[i])
        print('identifier',sort_median_users[i])
print('za',np.median(mosarray))
#calculate interquartile range for each user
iqr_ranges=[]
for user_score in users_scores:
    q3, q1 = np.percentile(user_score, [75,25])
    iqr = q3 - q1
    iqr_ranges.append(iqr)
min_iqr=np.min(iqr_ranges)
max_iqr=np.max(iqr_ranges)


#sorted users
group_size=6
right_order=index[0:group_size].tolist()+index[-group_size:].tolist()
#sort identifiers based on right_order
identifiers=[identifiers[i] for i in right_order]

#collect users score plus mos scores in one array
ordered_users=[]
for num in right_order[0:group_size]:
    ordered_users= ordered_users + users_scores[num].tolist()
ordered_users= ordered_users + mosarray.tolist()
for num in right_order[-group_size:]:
    ordered_users= ordered_users + users_scores[num].tolist()

#save users name for dataframe structure
#save users name for dataframe structure
users=[]
for u in right_order:
    for i in range(120):
        users.append('Z_'+str(u))
for i in range(120):
    users.append('Z')

#create datafram for boxplot
df = pd.DataFrame({'Users':users,'ordered':ordered_users})
df = df[['Users','ordered']]
dd=pd.melt(df,id_vars=['Users'],value_vars=['ordered'],var_name='')
my_colors = ["#ffd700"]

#plot boxplot
fig = plt.figure(figsize=(20, 10),dpi=100)
font = font_manager.FontProperties(size=50)
sns.set(rc={'figure.figsize':(8,3)},style="white")
sns.set_palette(my_colors)
ax=sns.boxplot(x='Users',y='value',data=dd,hue='')#,palette=my_pal
ax.tick_params(bottom=True)
ax.tick_params(left=True)
ax.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
plt.locator_params(axis='x', nbins=5)
ax.set(ylabel='QoE')
xlabels_group_one=['Z'+str(i+1) for i in range(group_size)]+['']
xlabels_group_two=['Z'+str(i+1) for i in range(len(isc)-group_size,len(isc))]
xlabels=xlabels_group_one+xlabels_group_two
ax.set_xticklabels(xlabels,fontsize=60,color='k')
ax.set_xticks([0, 5, 7, 12])
plt.yticks(fontsize=60, color='k')
plt.xticks(fontsize=60, color='k')
plt.ylabel('Score', fontsize=60,color='k')
plt.xlabel('Rater', fontsize=60, color='k')
plt.yticks([1, 20, 40, 60, 80, 100])
#add vertical line red dashed in middle of graph
#plt.axvline(x=6.5, color='r', linestyle='--',linewidth=5)
fig = ax.get_figure()
gold_patch = mpatches.Patch(color='#ffd700', label='Atypical raters')
green_patch = mpatches.Patch(color='#4b8521', label='ZA, average rater')
plt.legend(handles=[gold_patch,green_patch],loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=50, ncol=2, framealpha=0, columnspacing=0.5,handletextpad=0.4,prop=font)
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
plt.savefig('Fig_5a.png',bbox_inches='tight')
plt.close()

#from png to pdf high quality
from PIL import Image
im = Image.open('Fig_5a.png')
rgb_im = im.convert('RGB')
rgb_im.save('Fig_5a.pdf')
