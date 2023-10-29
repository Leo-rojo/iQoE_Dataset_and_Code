import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

colori=cm.get_cmap('tab10').colors

folder='../output_data/save_all_models_ave_users/'

font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 25,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 25,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 25}
plt.rc('font', **font_general)

#collect metrics for each models
sc=['maeb','maep','maev','maes','maesdn','maef','maeva','mael','maeiqoe','maeiqoeg']
mae_all_users=[]
maestd_all_users=[]
for score in sc:
    mae=[]
    maesd=[]
    for i in range(256):
        mae.append(np.load(folder+score+'_'+str(i)+'_maeave.npy'))
        maesd.append(np.load(folder+score+'_'+str(i)+'_maestd.npy'))
    mae_all_users.append(mae)
    maestd_all_users.append(maesd)

#calculate the average of the 10 models in mae_all_users
mae_all_users_ave=[]
for i in range(len(mae_all_users)):
    mae_all_users_ave.append(np.mean(mae_all_users[i]))

#p1203_and_LSTM_models
sota_aves_mae=[]
sota_stds_mae=[]
sota_aves_rmse=[]
sota_stds_rmse=[]

sc=['rmseb','rmsep','rmsev','rmses','rmsesdn','rmsef','rmseva','rmsel','rmseiqoe','rmseiqoeg']
rmse_all_users=[]
rmsestd_all_users=[]
for score in sc:
    rmse=[]
    rmsestd=[]
    for i in range(256):
        rmse.append(np.load(folder+score+'_'+str(i)+'_rmseave.npy'))
        rmsestd.append(np.load(folder+score+'_'+str(i)+'_rmsestd.npy'))
    rmse_all_users.append(rmse)
    rmsestd_all_users.append(rmsestd)

#calculate the average of the 10 models in rmse_all_users
rmse_all_users_ave=[]
for i in range(len(rmse_all_users)):
    rmse_all_users_ave.append(np.mean(rmse_all_users[i]))

#p1203_and_LSTM_models
for sota in ['p1203','bilstm']:
    sota_aves_mae.append(np.load('../output_data/sota_results/' + sota + '_scores/mae_ave.npy'))
    sota_stds_mae.append(np.load('../output_data/sota_results/' + sota + '_scores/mae_std.npy'))
    sota_aves_rmse.append(np.load('../output_data/sota_results/' + sota + '_scores/rmse_ave.npy'))
    sota_stds_rmse.append(np.load('../output_data/sota_results/' + sota + '_scores/rmse_std.npy'))

#calculate the average of sota_ave_mae for both arrays
sota_ave_mae=np.mean(sota_aves_mae,axis=1)
sota_ave_rmse=np.mean(sota_aves_rmse,axis=1)

mae_gains_factor=[]
for i in range(10):
    mae_gains_factor.append(mae_all_users_ave[i]/mae_all_users_ave[8])
mae_gains_factor.append(sota_ave_mae[0]/mae_all_users_ave[8])
mae_gains_factor.append(sota_ave_mae[1]/mae_all_users_ave[8])

rmse_gains_factor=[]
for i in range(10):
    rmse_gains_factor.append(rmse_all_users_ave[i]/rmse_all_users_ave[8])
rmse_gains_factor.append(sota_ave_rmse[0]/rmse_all_users_ave[8])
rmse_gains_factor.append(sota_ave_rmse[1]/rmse_all_users_ave[8])
not_sorted=['b','p','v','s','sdn','f','va','l','iqoe','iqoeg','p1203','bilstm']
#print(mae_gains_factor)
#print(rmse_gains_factor)
howsorted=['b', 'l', 'v', 'p', 's', 'f', 'va', 'sdn', 'p1203' ,'bilstm']
#order mae_gains_factor based on howsorted
mae_gains_factor_sorted=[]
rmse_gains_factor_sorted=[]
for i in howsorted:
    idx=not_sorted.index(i)
    mae_gains_factor_sorted.append(mae_gains_factor[idx])
    rmse_gains_factor_sorted.append(rmse_gains_factor[idx])
#print('sorted')
#print(mae_gains_factor_sorted)
#print(rmse_gains_factor_sorted)


###################names
qoe_model=['B','G','R','S','V','F','N','A']
qmr=np.repeat(qoe_model,32)
users_names=[]
for i in range(8):
    for k in range(32):
        users_names.append(qmr[32*i+k]+str(k+1))

###plot mae
#sort based on worst users for iqoep
values_from_worst_to_best=sorted(mae_all_users[0],reverse=True)
idx_by_difficult=[]
for i in values_from_worst_to_best:
    idx_by_difficult.append(mae_all_users[0].index(i))
names_by_difficulties=[users_names[i] for i in idx_by_difficult]

idx_most_difficult=[]
name_most_difficult=[]
for nam in ['B','G','R','S','V','F','N','A']:
    for c,i in enumerate(names_by_difficulties):
        if i[0]==nam:
            idx_most_difficult.append(idx_by_difficult[c])
            name_most_difficult.append(i)
            break

#the sort of mae_all_user ['maeb','maep','maev','maes','maesdn','maef','maeva','mael','maeiqoe','maeiqoeg']
worst_users_names = name_most_difficult  #[worst_users_names_[i] if i in [ 0,  3,  6,  9, 12, 15, 18, 21, 25] else "" for i in range(26)]
worst26_mosbit=[float(mae_all_users[0][i]) for i in idx_most_difficult]
worst26_mosbit_std=[float(maestd_all_users[0][i]) for i in idx_most_difficult]
worst26_moslogbit=[float(mae_all_users[7][i]) for i in idx_most_difficult]
worst26_moslogbit_std=[float(maestd_all_users[7][i]) for i in idx_most_difficult]
worst26_mosva=[float(mae_all_users[6][i]) for i in idx_most_difficult]
worst26_mosva_std=[float(maestd_all_users[6][i]) for i in idx_most_difficult]
worst26_iqoeg=[float(mae_all_users[-1][i]) for i in idx_most_difficult]
worst26_iqoeg_std=[float(maestd_all_users[-1][i]) for i in idx_most_difficult]
worst26_iqoep=[float(mae_all_users[-2][i]) for i in idx_most_difficult]
worst26_iqoep_std=[float(maestd_all_users[-2][i]) for i in idx_most_difficult]
worst26_mosftw=[float(mae_all_users[5][i]) for i in idx_most_difficult]
worst26_mosftw_std=[float(maestd_all_users[5][i]) for i in idx_most_difficult]
worst26_mossdn=[float(mae_all_users[4][i]) for i in idx_most_difficult]
worst26_mossdn_std=[float(maestd_all_users[4][i]) for i in idx_most_difficult]
worst26_mosssim=[float(mae_all_users[3][i]) for i in idx_most_difficult]
worst26_mosssim_std=[float(maestd_all_users[3][i]) for i in idx_most_difficult]
worst26_mospsnr=[float(mae_all_users[1][i]) for i in idx_most_difficult]
worst26_mospsnr_std=[float(maestd_all_users[1][i]) for i in idx_most_difficult]
worst26_mosvmaf=[float(mae_all_users[6][i]) for i in idx_most_difficult]
worst26_mosvmaf_std=[float(maestd_all_users[6][i]) for i in idx_most_difficult]
#sota
worst26_mosp1203=[sota_aves_mae[0][i] for i in idx_most_difficult]
worst26_mosp1203_std=[sota_stds_mae[0][i] for i in idx_most_difficult]
worst26_mosbilstm=[sota_aves_mae[1][i] for i in idx_most_difficult]
worst26_mosbilstm_std=[sota_stds_mae[1][i] for i in idx_most_difficult]
#save results
# np.save('./store_results/mosva',worst26_mosva)
# np.save('./store_results/iqoeg',worst26_iqoeg)
# np.save('./store_results/iqoep',worst26_iqoep)
# np.save('./store_results/mossdn',worst26_mossdn)
# np.save('./store_results/mospsnr',worst26_mospsnr)
# np.save('./store_results/mosssim',worst26_mosssim)
# np.save('./store_results/mosvmaf',worst26_mosvmaf)
# np.save('./store_results/mosftw',worst26_mosftw)

#plots mae histogram
for metric in ['mae']:
    fig = plt.figure(figsize=(20, 5),dpi=100)
    #plt.axhline(y=250, color='black', linestyle='-')
    barWidth = 1.65
    a = np.arange(0, 8 * 22, 22)
    b = [i + barWidth for i in a]
    c = [i + barWidth for i in b]
    d = [i + barWidth for i in c]
    e = [i + barWidth for i in d]
    f = [i + barWidth for i in e]
    g = [i + barWidth for i in f]
    h = [i + barWidth for i in g]
    l = [i + barWidth for i in h]
    m = [i + barWidth for i in l]
    n = [i + barWidth for i in m]
    o = [i + barWidth for i in n]
    plt.bar(a, worst26_iqoep, color='r', width=barWidth, linewidth=0, label='iQoE-personal_50q', align='edge',yerr=worst26_iqoep_std)
    plt.bar(b, worst26_mosbit, color=colori[1], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_mosbit_std)
    plt.bar(c, worst26_moslogbit, color=colori[2], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_moslogbit_std)
    plt.bar(d, worst26_mosvmaf, color=colori[8], width=barWidth, linewidth=0, label='VMAF-group', align='edge',yerr=worst26_mosvmaf_std)
    plt.bar(e, worst26_mospsnr, color=colori[7], width=barWidth, linewidth=0, label='PSNR-group', align='edge',yerr=worst26_mospsnr_std)
    plt.bar(f, worst26_mosssim, color=colori[6], width=barWidth, linewidth=0, label='SSIM-group', align='edge',yerr=worst26_mosssim_std)
    plt.bar(g, worst26_mosftw, color=colori[9], width=barWidth, linewidth=0, label='FTW-group', align='edge',yerr=worst26_mosftw_std)
    plt.bar(h, worst26_mosva, color=colori[4], width=barWidth, linewidth=0, label='VA-group', align='edge',yerr=worst26_mosva_std)
    plt.bar(l, worst26_mossdn, color='gold', width=barWidth, linewidth=0, label='SDN-group', align='edge',yerr=worst26_mossdn_std)
    plt.bar(m, worst26_mosp1203, color='darkblue', width=barWidth, linewidth=0, label='P1203', align='edge',yerr=worst26_mosp1203_std)
    plt.bar(n, worst26_mosbilstm, color=colori[5], width=barWidth, linewidth=0, label='BiLSTM', align='edge',yerr=worst26_mosbilstm_std)
    plt.bar(o, worst26_iqoeg, color=colori[0], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_iqoeg_std)
    # plt.xticks(np.arange(8) + barWidth*2.1, ['Bit','Log','Ps','Ss','Vm','FTW','SDN','VA'])
    plt.xlabel("Rater", fontdict=font_axes_titles)
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 65, 10))
    worst_26_names = []
    plt.xticks(np.arange(0, 8 * 22, 22) + barWidth * 6, worst_users_names, fontsize=25)
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.tick_params(axis='y', which='major', width=3, length=10)
    ax.set_ylim([0, 65])
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    # plt.title('treshold metric '+metric +' '+regr_choosen , fontdict=font_title)
    # plt.legend(fontsize=23,frameon=False,loc='upper left')
    plt.savefig('./histogram_' + metric + '.pdf', bbox_inches='tight')
    plt.close()

###plot rmse
#take 26 worst users for iQoE-p ['maeb','maep','maev','maes','maesdn','maef','maeva','mael','maeiqoe','maeiqoeg']
worst26_moslogbit_std_rmse=[rmsestd_all_users[7][i] for i in idx_most_difficult]
worst26_mosbit_rmse=[rmse_all_users[0][i] for i in idx_most_difficult]
worst26_mosbit_std_rmse=[rmsestd_all_users[0][i] for i in idx_most_difficult]
worst26_moslogbit_rmse=[rmse_all_users[7][i] for i in idx_most_difficult]
worst26_mosva_rmse=[rmse_all_users[6][i] for i in idx_most_difficult]
worst26_mosva_std_rmse=[rmsestd_all_users[6][i] for i in idx_most_difficult]
worst26_iqoeg_rmse=[rmse_all_users[-1][i] for i in idx_most_difficult]
worst26_iqoeg_std_rmse=[rmsestd_all_users[-1][i] for i in idx_most_difficult]
worst26_iqoep_rmse=[rmse_all_users[-2][i] for i in idx_most_difficult]
worst26_iqoep_std_rmse=[rmsestd_all_users[-2][i] for i in idx_most_difficult]
worst26_mosftw_rmse=[rmse_all_users[5][i] for i in idx_most_difficult]
worst26_mosftw_std_rmse=[rmsestd_all_users[5][i] for i in idx_most_difficult]
worst26_mossdn_rmse=[rmse_all_users[4][i] for i in idx_most_difficult]
worst26_mossdn_std_rmse=[rmsestd_all_users[4][i] for i in idx_most_difficult]
worst26_mosssim_rmse=[rmse_all_users[3][i] for i in idx_most_difficult]
worst26_mosssim_std_rmse=[rmsestd_all_users[3][i] for i in idx_most_difficult]
worst26_mospsnr_rmse=[rmse_all_users[1][i] for i in idx_most_difficult]
worst26_mospsnr_std_rmse=[rmsestd_all_users[1][i] for i in idx_most_difficult]
worst26_mosvmaf_rmse=[rmse_all_users[2][i] for i in idx_most_difficult]
worst26_mosvmaf_std_rmse=[rmsestd_all_users[2][i] for i in idx_most_difficult]
#sota
worst26_mosp1203_rmse=[sota_aves_rmse[0][i] for i in idx_most_difficult]
worst26_mosp1203_std_rmse=[sota_stds_rmse[0][i] for i in idx_most_difficult]
worst26_mosbilstm_rmse=[sota_aves_rmse[1][i] for i in idx_most_difficult]
worst26_mosbilstm_std_rmse=[sota_stds_rmse[1][i] for i in idx_most_difficult]
# np.save('./store_results/mosva_rmse',worst26_mosva_rmse)
# np.save('./store_results/iqoeg_rmse',worst26_iqoeg_rmse)
# np.save('./store_results/iqoep_rmse',worst26_iqoep_rmse)
# np.save('./store_results/mossdn_rmse',worst26_mossdn_rmse)
# np.save('./store_results/mospsnr_rmse',worst26_mospsnr_rmse)
# np.save('./store_results/mosssim_rmse',worst26_mosssim_rmse)
# np.save('./store_results/mosvmaf_rmse',worst26_mosvmaf_rmse)
# np.save('./store_results/mosftw_rmse',worst26_mosftw_rmse)
for metric in ['rmse']:
    fig = plt.figure(figsize=(20, 5), dpi=100)
    # plt.axhline(y=250, color='black', linestyle='-')
    barWidth = 1.65
    a = np.arange(0, 8 * 22, 22)
    b = [i + barWidth for i in a]
    c = [i + barWidth for i in b]
    d = [i + barWidth for i in c]
    e = [i + barWidth for i in d]
    f = [i + barWidth for i in e]
    g = [i + barWidth for i in f]
    h = [i + barWidth for i in g]
    l = [i + barWidth for i in h]
    m = [i + barWidth for i in l]#qoe_model=['B','L','V','P','S',,'F','A','N']
    n = [i + barWidth for i in m]
    o = [i + barWidth for i in n]
    # plt.bar(a, worst26_mosbit, color='c', width=barWidth - 0.1, linewidth=0, label='Bit-group',align='edge')  # yerr=ss_elab_std[2]
    plt.bar(a, worst26_iqoep_rmse, color='r', width=barWidth, linewidth=0, label='iQoE-personal_50q', align='edge',yerr=worst26_iqoep_std_rmse)
    plt.bar(b, worst26_mosbit_rmse, color=colori[1], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_mosbit_std_rmse)
    plt.bar(c, worst26_moslogbit_rmse, color=colori[2], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_moslogbit_std_rmse)
    plt.bar(d, worst26_mosvmaf_rmse, color=colori[8], width=barWidth, linewidth=0, label='VMAF-group', align='edge',
            yerr=worst26_mosvmaf_std_rmse)
    plt.bar(e, worst26_mospsnr_rmse, color=colori[7], width=barWidth, linewidth=0, label='PSNR-group', align='edge',
            yerr=worst26_mospsnr_std_rmse)
    plt.bar(f, worst26_mosssim_rmse, color=colori[6], width=barWidth, linewidth=0, label='SSIM-group', align='edge',
            yerr=worst26_mosssim_std_rmse)
    plt.bar(g, worst26_mosftw_rmse, color=colori[9], width=barWidth, linewidth=0, label='FTW-group', align='edge',
            yerr=worst26_mosftw_std_rmse)
    plt.bar(h, worst26_mosva_rmse, color=colori[4], width=barWidth, linewidth=0, label='VA-group', align='edge',
            yerr=worst26_mosva_std_rmse)
    plt.bar(l, worst26_mossdn_rmse, color='gold', width=barWidth, linewidth=0, label='SDN-group', align='edge',
            yerr=worst26_mossdn_std_rmse)
    plt.bar(m, worst26_mosp1203_rmse, color='darkblue', width=barWidth, linewidth=0, label='P1203-group', align='edge',
            yerr=worst26_mosp1203_std_rmse)
    plt.bar(n, worst26_mosbilstm_rmse, color=colori[5], width=barWidth, linewidth=0, label='BiLSTM-group', align='edge',
            yerr=worst26_mosbilstm_std_rmse)
    plt.bar(o, worst26_iqoeg_rmse, color=colori[0], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_iqoeg_std_rmse)

    # plt.xticks(np.arange(8) + barWidth*2.1, ['Bit','Log','Ps','Ss','Vm','FTW','SDN','VA'])
    plt.xlabel("Rater", fontdict=font_axes_titles)
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 65, 10))
    worst_26_names = []
    plt.xticks(np.arange(0, 8*22, 22) + barWidth*6, worst_users_names,fontsize=25)
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.tick_params(axis='y', which='major', width=3, length=10)
    ax.set_ylim([0, 65])
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    # plt.title('treshold metric '+metric +' '+regr_choosen , fontdict=font_title)
    # plt.legend(fontsize=23,frameon=False,loc='upper left')
    plt.savefig('../histogram_' + metric + '.pdf', bbox_inches='tight')
    plt.close()