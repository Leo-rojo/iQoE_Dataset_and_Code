import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
from matplotlib.patches import PathPatch
import statsmodels.api as sm
import matplotlib.lines as mlines
#insert path to your folder Figure 15
isc=[]
identifiers=[]
#collect individual scores and mos
for fold in os.listdir('../output_data/dataset'):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        print(identifier)
        identifiers.append(identifier)
        #collect scores
        xls = pd.ExcelFile(r"../output_data/dataset/user_"+identifier) #use r before absolute file path
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
#collect data for description
sorted_for_data=aveeachuserarray[index]
#calculate interquartile range for each user
iqr_ranges=[]
for user_score in users_scores:
    q3, q1 = np.percentile(user_score, [75,25])
    iqr = q3 - q1
    iqr_ranges.append(iqr)
min_iqr=np.min(iqr_ranges)
max_iqr=np.max(iqr_ranges)


#sorted raters
right_order=index
#sort identifiers based on right_order
identifiers=[identifiers[i] for i in right_order]

#collect raters score plus mos scores in one array
ordered_users=[]
for num in right_order:
    ordered_users= ordered_users + users_scores[num].tolist()

ordered_users= ordered_users + mosarray.tolist()

#save raters name for dataframe structure
#save raters name for dataframe structure
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

#sort raters scores
ordered_users=[]
for num in right_order:
    ordered_users.append(users_scores[num].tolist())
#ECDF for 4users and average user
lab=['Z1','Z6','Z115','Z120','ZA']
my_pal = [ "b", 'grey','m',"c","#4b8521"]



fig=plt.figure(figsize=(20, 10),dpi=100)
subsec=[]
for i in range(6):
    subsec.append(ordered_users[i])
for i in range(len(ordered_users)-6,len(ordered_users)):
    subsec.append(ordered_users[i])
subsec.append(mosarray)
style=['-','-.',':','--','-']
#different shades of the same color
my_pal_blue = sns.color_palette("Blues", 12)
my_pal_red = sns.color_palette("Reds", 12)

for i in range(len(subsec)):
    ecdf = sm.distributions.ECDF(subsec[i])
    if i == len(subsec) - 1:
        plt.step(ecdf.x, ecdf.y,linewidth=6.0,color='g',linestyle='-')
    elif i<6:
        plt.step(ecdf.x, ecdf.y, linewidth=6.0,color=my_pal_blue[i+3],linestyle='dotted')#,linestyle=style[i]),color=my_pal[i],label=lab[i],
    else:
        plt.step(ecdf.x, ecdf.y, linewidth=6.0,color=my_pal_red[i-6+3],linestyle='dashed')
plt.margins(0,0)
plt.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
plt.ylabel('% of scores', fontsize=55)
plt.xticks(fontsize=55)
plt.yticks(fontsize=55)
plt.xlabel('Score', fontsize=55)
plt.xticks([1,20,40,60,80,100])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['0','20','40','60','80','100'])
HA_line = mlines.Line2D([], [], color='#4b8521', linestyle='-', linewidth=7, label='ZA')
H1_line = mlines.Line2D([], [], color=my_pal_blue[5], linestyle='dotted', linewidth=7, label='Z1-Z6')
H32_line = mlines.Line2D([], [], color=my_pal_red[5], linestyle='dashed', linewidth=7, label='Z115-Z120')
#H2_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=2, label='Z2')
#H31_line = mlines.Line2D([], [], color='m', linestyle='--', linewidth=2, label='Z129')
font = font_manager.FontProperties(size=50)
handles=[H1_line,H32_line,HA_line]
plt.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=50, ncol=5, framealpha=0, columnspacing=0.5,handletextpad=0.4,prop=font)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('../Figure 15b.pdf',bbox_inches='tight')

