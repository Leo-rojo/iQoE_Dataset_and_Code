import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import linear_model
from scipy.optimize import curve_fit
import warnings
#insert path to your folder Figure 7
warnings.filterwarnings("ignore")
from matplotlib import cm
def find_divisors(end_users):
    divisors = []
    for i in range(1, end_users + 1):
        if end_users % i == 0:
            divisors.append(i)
    return divisors
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
plt.rc('font', **font_general)

print(find_divisors(128))
group_division=[1, 2, 4, 8, 16, 32, 64, 128]
start_users=0
end_users=128
def fit_linear(all_features, users_scores):
    # multi-linear model fitting
    X = all_features
    y = users_scores

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y)

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]
def fit_nonlinear(all_features, users_scores):
    def fun(data, a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, users_scores, maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d
def fit_supreg(all_features, users_scores):
    data = np.array(all_features)
    target = np.array(users_scores)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_

#split in groups
baselines_all_users_score=np.load('input_data/baselines_scores.npy')
test_scores_all_users=np.load('input_data/test_scores.npy')
#merge them
baselines_all_users_score_total=[]
for i in range(len(baselines_all_users_score)):
    baselines_all_users_score_total.append(baselines_all_users_score[i].tolist()+test_scores_all_users[i].tolist())


mosarray=np.mean(baselines_all_users_score_total,axis=0)
aveeachuserarray=np.median(baselines_all_users_score_total,axis=1)
index=np.argsort(aveeachuserarray)

user_considered=index[-1]
identifiers=np.load('input_data/identifiers_order.npy')






print('user considered',identifiers[user_considered])
for i in range(len(index)):
    print(identifiers[index[i]],aveeachuserarray[index[i]])

user_partecipating_in_grouping=index[start_users:end_users]

#divide array in n_group split
maes_all=[]
rmses_all=[]
for group_n in group_division:#[1,2,4,8,16,32]:
    print('group_n',group_n)
    splitted=np.array_split(user_partecipating_in_grouping, group_n)
    splitted_for_calculating_mosdistances=np.array_split(np.array(baselines_all_users_score_total)[user_partecipating_in_grouping], group_n)
    splitted_for_calculating_Mos=np.array_split(baselines_all_users_score[user_partecipating_in_grouping], group_n)
    mos_each_split=[]
    for i in range(group_n):
        mos_each_split.append(np.mean(splitted_for_calculating_Mos[i],axis=0))
    medianmos_each_split=[np.median(i) for i in np.mean(splitted_for_calculating_mosdistances,axis=1)]
    median_considered_user=np.median(baselines_all_users_score[user_considered])
    #index of medianmos_each_split closest to median_considered_user
    index_of_medianmos_each_split_closest_to_median_considered_user=np.argmin(np.abs(np.array(medianmos_each_split)-median_considered_user))





    mos_for_training=mos_each_split[index_of_medianmos_each_split_closest_to_median_considered_user]
    #take iqoe of the user considered (last one of the index)
    iqoes_mae=np.load('input_data/iQoE_mae_each_user.npy')[user_considered]
    iqoes_rmse=np.load('input_data/iQoE_rmse_each_user.npy')[user_considered]
    #take p1203 of the user considered (last one of the index)
    p1203_mae=np.load('input_data/p1203_mae_each_user.npy')[user_considered]
    p1203_rmse=np.load('input_data/p1203_rmse_each_user.npy')[user_considered]
    #take lstm of the user considered (last one of the index)
    lstm_mae=np.load('input_data/lstm_mae_each_user.npy')[user_considered]
    lstm_rmse=np.load('input_data/lstm_rmse_each_user.npy')[user_considered]

    #ssim
    x_tr=np.load('input_data/features_qoes_train/feat_ssim.npy')
    x_test=np.load('input_data/features_qoes_test/feat_ssim.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')

    y=mos_for_training
    a,b,c=fit_linear(x_tr,y)
    #calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_ssim=mean_absolute_error(y_test, y_pred)
    rmse_ssim=sqrt(mean_squared_error(y_test, y_pred))

    #ftw
    x_tr=np.load('input_data/features_qoes_train/feat_ftw.npy')
    x_test=np.load('input_data/features_qoes_test/feat_ftw.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y = mos_for_training
    a, b, c, d = fit_nonlinear((x_tr[:, 0], x_tr[:, 1]), y)
    y_pred = a * np.exp(-(b * x_test[:, 0] + c) * x_test[:, 1]) + d
    y_pred[y_pred > 100] = 100
    #calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    mae_ftw=mean_absolute_error(y_test, y_pred)
    rmse_ftw=sqrt(mean_squared_error(y_test, y_pred))

    #videoatlas
    x_tr=np.load('input_data/features_qoes_train/feat_va.npy')
    x_test=np.load('input_data/features_qoes_test/feat_va.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    reg=fit_supreg(x_tr, y)
    y_test = y_test_all_users[user_considered]
    y_pred = reg.predict(x_test)
    mae_videoatlas=mean_absolute_error(y_test, y_pred)
    rmse_videoatlas=sqrt(mean_squared_error(y_test, y_pred))

    #psnr
    x_tr=np.load('input_data/features_qoes_train/feat_psnr.npy')
    x_test=np.load('input_data/features_qoes_test/feat_psnr.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    a, b, c = fit_linear(x_tr, y)
    # calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_psnr=mean_absolute_error(y_test, y_pred)
    rmse_psnr=sqrt(mean_squared_error(y_test, y_pred))

    #bitrate
    x_tr=np.load('input_data/features_qoes_train/feat_bit.npy')
    x_test=np.load('input_data/features_qoes_test/feat_bit.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    a, b, c = fit_linear(x_tr, y)
    # calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_bit=mean_absolute_error(y_test, y_pred)
    rmse_bit=sqrt(mean_squared_error(y_test, y_pred))

    #logbit
    x_tr=np.load('input_data/features_qoes_train/feat_logbit.npy')
    x_test=np.load('input_data/features_qoes_test/feat_logbit.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    a, b, c = fit_linear(x_tr, y)
    # calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_logbit=mean_absolute_error(y_test, y_pred)
    rmse_logbit=sqrt(mean_squared_error(y_test, y_pred))

    #vmaf
    x_tr=np.load('input_data/features_qoes_train/feat_vmaf.npy')
    x_test=np.load('input_data/features_qoes_test/feat_vmaf.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    a, b, c = fit_linear(x_tr, y)
    # calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_vmaf=mean_absolute_error(y_test, y_pred)
    rmse_vmaf=sqrt(mean_squared_error(y_test, y_pred))

    #sdn
    x_tr=np.load('input_data/features_qoes_train/feat_sdn.npy')
    x_test=np.load('input_data/features_qoes_test/feat_sdn.npy')
    y_test_all_users=np.load('input_data/test_scores.npy')
    y=mos_for_training
    a, b, c = fit_linear(x_tr, y)
    # calculate mae and rmse of the model for the user in index
    y_test = y_test_all_users[user_considered]
    y_pred = a * x_test[:, 0] + b * x_test[:, 1] + c * x_test[:, 2]
    mae_sdn=mean_absolute_error(y_test, y_pred)
    rmse_sdn=sqrt(mean_squared_error(y_test, y_pred))


    # print('group_n',group_n)
    # print('mae_ssim',mae_ssim)
    # print('mae_ftw',mae_ftw)
    # print('mae_videoatlas',mae_videoatlas)
    # print('mae_psnr',mae_psnr)
    # print('rmse_ssim',rmse_ssim)
    # print('rmse_ftw',rmse_ftw)
    # print('rmse_videoatlas',rmse_videoatlas)
    # print('rmse_psnr',rmse_psnr)
    maes_all.append([iqoes_mae,mae_bit,mae_logbit,mae_videoatlas,mae_ssim,mae_psnr,mae_vmaf,mae_ftw,mae_sdn,p1203_mae,lstm_mae])
    rmses_all.append([iqoes_rmse,rmse_bit,rmse_logbit,rmse_videoatlas,rmse_ssim,rmse_psnr,rmse_vmaf,rmse_ftw,rmse_sdn,p1203_rmse,lstm_rmse])


#barplot maes_all
names=['iqoes','bit','logbit','videoatlas','ssim','psnr','vmaf','ftw','sdn','p1203','lstm']
bymodels=[]
for model in range(11):
    bymodel=[]
    for i in range(8):
        bymodel.append(maes_all[i][model])
    bymodels.append(bymodel)


fig = plt.figure(figsize=(20, 10),dpi=100)
conta = 0
markers= ['o', 's', 'D', 'v', '*', 'h', '^', '8', 'P', '<', 'X']
stile = ['-', '--', '-.', ':', '-', '--', '-','-.', ':','-.', ':']
col=['r',colori[1],colori[2],colori[4],colori[6],colori[7],colori[8],colori[9],'gold','darkblue',colori[5]]
for regr in bymodels:
    #print(regr)
    plt.plot(regr,stile[conta], linewidth='7',color=col[conta],marker=markers[conta],markersize=25,markeredgecolor='black',zorder=len(bymodels) - conta)
    conta += 1
#plt.grid()
plt.xlabel("Number of reference groups", fontdict=font_axes_titles)
plt.xticks([i for i in range(8)],[str(i) for i in group_division] )#1,5,10,20,50,100
plt.ylabel('MAE', fontdict=font_axes_titles)
#plt.yticks(range(0, 20, 2))
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.yticks(np.arange(0, 81, 10))
plt.ylim(2, 70)
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
plt.savefig('group_reduction_mae_lines.pdf',bbox_inches='tight',)
plt.close()

#put bymodels in datafram with columns and rows
import pandas as pd
df = pd.DataFrame(bymodels, columns=[str(i) for i in group_division], index=names)
#save to excel
df.to_excel('./mae_different_group_lines.xlsx')


#barplot rmses_all
names=['iqoes','bit','logbit','videoatlas','ssim','psnr','vmaf','ftw','sdn','p1203','lstm']
bymodels=[]
for model in range(11):
    bymodel=[]
    for i in range(8):
        bymodel.append(rmses_all[i][model])
    bymodels.append(bymodel)

fig = plt.figure(figsize=(20, 10),dpi=100)
conta = 0
markers= ['o', 's', 'D', 'v', '*', 'h', '^', '8', 'P', '<', 'X']
stile = ['-', '--', '-.', ':', '-', '--', '-','-.', ':','-.', ':']
col=['r',colori[1],colori[2],colori[4],colori[6],colori[7],colori[8],colori[9],'gold','darkblue',colori[5]]
for regr in bymodels:
    #print(regr)
    plt.plot(regr,stile[conta], linewidth='7',color=col[conta],marker=markers[conta],markersize=25,markeredgecolor='black',zorder=len(bymodels) - conta)
    conta += 1
#plt.grid()
plt.xlabel("Number of reference groups", fontdict=font_axes_titles)
plt.xticks([i for i in range(8)],[str(i) for i in group_division] )
plt.ylabel('RMSE', fontdict=font_axes_titles)
#plt.yticks(range(0, 20, 2))
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.yticks(np.arange(0, 81, 10))
plt.ylim(2, 70)
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
plt.savefig('group_reduction_rmse_lines.pdf',bbox_inches='tight',)
plt.close()

#put bymodels in datafram with columns and rows
import pandas as pd
df = pd.DataFrame(bymodels, columns=[str(i) for i in group_division], index=names)
#save to excel
df.to_excel('./rmse_different_group_lines.xlsx')




