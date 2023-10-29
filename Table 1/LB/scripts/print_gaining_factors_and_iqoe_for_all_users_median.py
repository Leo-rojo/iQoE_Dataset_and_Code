import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import jensenshannon
import math
#insert path to your folder Figure 6 and 8 and Table 1
colori=cm.get_cmap('tab10').colors

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


leg = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas','p1203', 'LSTM']
metric='mae'
colors=[colori[1],colori[2],colori[7],colori[6],colori[8],colori[9],'gold',colori[4],'lightblue',colori[5]]
regr_choosen='SVR'
#load data
group_maes=np.load('../input_data/group_mae_each_user.npy')
group_rmses=np.load('../input_data/group_rmse_each_user.npy')
pers_maes=np.load('../input_data/personalzed_literature_qoe_mae_each_user.npy')
pers_rmses=np.load('../input_data/peronalized_listerature_qoe_rmse_each_user.npy')
iqoe_maes=np.load('../input_data/iQoE_mae_each_user.npy')
iqoe_rmses=np.load('../input_data/iQoE_rmse_each_user.npy')
p1203_maes=np.load('../input_data/p1203_mae_each_user.npy')
p1203_rmses=np.load('../input_data/p1203_rmse_each_user.npy')
lstm_maes=np.load('../input_data/lstm_mae_each_user.npy')
lstm_rmses=np.load('../input_data/lstm_rmse_each_user.npy')
nr_users=group_maes.shape[0]

path_iQoE='../input_data/users'
identifier_ordered_by_name=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        identifier_ordered_by_name.append(identifier)
#print(identifier_ordered_by_name)

gain_factors_by_users_10models=[]
iqoe_alone=[]
for i in range(nr_users):
    iqoe_alone.append(iqoe_maes[i])
    gain_factors=[]
    #print('user',identifier_ordered_by_name[i])
    for each_model in group_maes[i]:
        gain_factors.append(each_model / iqoe_maes[i])
    gain_factors.append(p1203_maes[i] / iqoe_maes[i])
    gain_factors.append(lstm_maes[i] / iqoe_maes[i])
    gain_factors_by_users_10models.append(gain_factors)
    #print(gain_factors)

#gain_factors_by_users_10models average
gain_factors_by_users_10models_ave=np.array(gain_factors_by_users_10models)
gain_factors_by_users_10models_ave=np.mean(gain_factors_by_users_10models,axis=1)



#############sort user based on their median as in user heterogeneity plot
isc=[]
id_array=[]
#collect individual scores and mos
folder_xls='../input_data/dataset'
for fold in os.listdir(folder_xls):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        #print(identifier)
        #collect scores
        xls = pd.ExcelFile(folder_xls+"/user_"+identifier) #use r before absolute file path
        sheetX = xls.parse(0)
        iarray=sheetX['score'].tolist()
        isc.append(iarray)
        id_array.append(identifier)

mosarray=np.mean(isc,axis=0)
collect_all=[]
users_scores=np.array(isc).reshape(len(isc),120)

#distance median
medianeachuserarray=np.median(isc,axis=1)
medianmos=np.median(mosarray)
distancefrommedian=np.abs(medianeachuserarray-medianmos)
#sort distancefrommedian and take the first 4
index=np.argsort(distancefrommedian)#[0:4]





#sort gain_factors_by_users_10models based on index
gain_factors_by_users_10models_sortedbymedian=np.array(gain_factors_by_users_10models)
gain_factors_by_users_10models_sortedbymedian=gain_factors_by_users_10models_sortedbymedian[index]
#sort identifier_ordered_by_name based on index
identifier_ordered_by_name_sortedbymedian=np.array(identifier_ordered_by_name)
identifier_ordered_by_name_sortedbymedian=identifier_ordered_by_name_sortedbymedian[index]
#sort iqoe_alone based on index
iqoe_alone=np.array(iqoe_alone)
iqoe_alone=iqoe_alone[index]
#sort gain_factors_by_users_10models_ave based on index
gain_factors_by_users_10models_ave_sortedbymedian=np.array(gain_factors_by_users_10models_ave)
gain_factors_by_users_10models_ave_sortedbymedian=gain_factors_by_users_10models_ave_sortedbymedian[index]

#put each element of gain factors as column of dataframe and save it as excel
df=pd.DataFrame(gain_factors_by_users_10models_sortedbymedian,columns=leg)
#add gain_factors_by_users_10models_ave as column
df['average']=gain_factors_by_users_10models_ave_sortedbymedian
#add iqoe alone as column
df['iqoe mae']=iqoe_alone
# use identifier_ordered_by_name as index
df.index=identifier_ordered_by_name_sortedbymedian
#add average as row
df.loc['average']=df.mean()

#add average of last 10% rows as row
ten_percent=math.ceil(len(id_array)*10/100)
df.loc['average of last 10%']=df.iloc[-(ten_percent):].mean()
df = df.applymap("{:.2f}".format)
df.to_excel('../output_data/gain_factors_mae_all_users_median.xlsx')

#################################################################the same but for rmse
gain_factors_by_users_10models_rmse=[]
iqoe_alone_rmse=[]
for i in range(nr_users):
    iqoe_alone_rmse.append(iqoe_rmses[i])
    gain_factors_rmse=[]
    #print('user',identifier_ordered_by_name[i])
    for each_model in group_rmses[i]:
        gain_factors_rmse.append(each_model / iqoe_rmses[i])
    gain_factors_rmse.append(p1203_rmses[i] / iqoe_rmses[i])
    gain_factors_rmse.append(lstm_rmses[i] / iqoe_rmses[i])
    gain_factors_by_users_10models_rmse.append(gain_factors_rmse)
    #print(gain_factors_rmse)

#gain_factors_by_users_10models average
gain_factors_by_users_10models_ave_rmse=np.array(gain_factors_by_users_10models_rmse)
gain_factors_by_users_10models_ave_rmse=np.mean(gain_factors_by_users_10models_rmse,axis=1)

#sort gain_factors_by_users_10models based on index
gain_factors_by_users_10models_sortedbymedian_rmse=np.array(gain_factors_by_users_10models_rmse)
gain_factors_by_users_10models_sortedbymedian_rmse=gain_factors_by_users_10models_sortedbymedian_rmse[index]

#sort iqoe_alone based on index
iqoe_alone_rmse=np.array(iqoe_alone_rmse)
iqoe_alone_rmse=iqoe_alone_rmse[index]
#sort gain_factors_by_users_10models_ave based on index
gain_factors_by_users_10models_ave_sortedbymedian_rmse=np.array(gain_factors_by_users_10models_ave_rmse)
gain_factors_by_users_10models_ave_sortedbymedian_rmse=gain_factors_by_users_10models_ave_sortedbymedian_rmse[index]

#put each element of gain factors as column of dataframe and save it as excel
df_rmse=pd.DataFrame(gain_factors_by_users_10models_sortedbymedian_rmse,columns=leg)
#add gain_factors_by_users_10models_ave as column
df_rmse['average']=gain_factors_by_users_10models_ave_sortedbymedian_rmse
#add iqoe alone as column
df_rmse['iqoe mae']=iqoe_alone_rmse

# use identifier_ordered_by_name as index
df_rmse.index=identifier_ordered_by_name_sortedbymedian
#add average as row
df_rmse.loc['average']=df_rmse.mean()
#add average of last 4 rows as row
df_rmse.loc['average of last 10%']=df_rmse.iloc[-(ten_percent):].mean()
df_rmse = df_rmse.applymap("{:.2f}".format)
df_rmse.to_excel('../output_data/gain_factors_rmse_all_users_median.xlsx')









