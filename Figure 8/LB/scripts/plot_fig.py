import statsmodels.api as sm
#for ML model
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

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

#stile = ['--', '-.', ':', '-', '--', '-','-.', ':','-.', ':']
#col=[colori[1],colori[2],colori[4],colori[6],colori[7],colori[8],colori[9],'gold','lightblue',colori[5]]
#for regr in [rmse_bitrate,rmse_logbitrate,rmse_videoatlas,rmse_ssim,rmse_psnr,rmse_vmaf,rmse_ftw,rmse_sdn,[p1203_rmse for i in range(len(rmse_ssim))],[lstm_rmse for i in range(len(rmse_ssim))]]:

stile= ['--','-.','--','-','-','-.',':',':','-.',':']
leg = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas','p1203', 'LSTM']
metric='mae'
colors=[colori[1],colori[2],colori[7],colori[6],colori[8],colori[9],'gold',colori[4],'darkblue',colori[5]]
regr_choosen='SVR'
#load data
group_maes=np.load('../output_data/group_mae_each_user.npy')
group_rmses=np.load('../output_data/group_rmse_each_user.npy')
iqoe_maes=np.load('../output_data/iQoE_mae_each_user.npy')
iqoe_rmses=np.load('../output_data/iQoE_rmse_each_user.npy')
p1203_maes=np.load('../output_data/p1203_mae_each_user.npy')
p1203_rmses=np.load('../output_data/p1203_rmse_each_user.npy')
lstm_maes=np.load('../output_data/lstm_mae_each_user.npy')
lstm_rmses=np.load('../output_data/lstm_rmse_each_user.npy')
pers_maes=np.load('../output_data/personalized_literature_qoe_mae_each_user.npy')
pers_rmses=np.load('../output_data/personalized_literature_qoe_rmse_each_user.npy')

nr_users=group_maes.shape[0]


#iqoe vs personalized
for metric in ['mae','rmse']:
    main_path_for_save_fig = '../'

    #np.save(main_path_for_save_fig + '/' + metric + 'values_ave', final_ave)
    #np.save(main_path_for_save_fig + '/' + metric + 'values_std', final_std)
    save_distributions=[]
    fig = plt.figure(figsize=(20, 10), dpi=100)

    #stile = [ '-','--']

    if metric=='mae':
        pers_dist=pers_maes.transpose()
        pers_dist=pers_dist.tolist()
        pers_dist.append(p1203_maes)
        pers_dist.append(lstm_maes)
    else:
        pers_dist=pers_rmses.transpose()
        pers_dist=pers_dist.tolist()
        pers_dist.append(p1203_rmses)
        pers_dist.append(lstm_rmses)

    conta = 0
    for ss in pers_dist:
        #scores=np.load('results_collection_rigsvsrnd/'+ss+'_'+metric+'_each_user.npy')
        #print(scores)
        ecdf = sm.distributions.ECDF(ss)
        save_distributions.append(ecdf)
        plt.step(ecdf.x, ecdf.y, label=leg[conta], linewidth=7.0,color=colors[conta], linestyle=stile[conta])
        conta+=1

    #add iQoE
    if metric=='mae':
        ecdf = sm.distributions.ECDF(iqoe_maes)
    else:
        ecdf = sm.distributions.ECDF(iqoe_rmses)
    save_distributions.append(ecdf)
    plt.step(ecdf.x, ecdf.y, label='iqoe', linewidth=7.0, color='r', linestyle='-')

    np.save('../output_data'+'/'+'values_ecdf_pers'+metric,save_distributions)

    plt.xlabel(metric.upper(), fontdict=font_axes_titles)
    plt.ylabel('% of raters', fontdict=font_axes_titles)
    #plt.xticks(np.arange(0,20,5))
    # plt.title('ECDF',fontdict=font_title)
    # lege = [leg[-1],leg[0]]
    # colorsi = [colors[-1], colors[0]]
    # handles = [
    #     Patch(facecolor=color, label=label)
    #     for label, color in zip(lege, colorsi)
    # ]
    # plt.legend(ncol=2, frameon=False, handles=handles, handlelength=2., loc='lower right',handleheight=0.7,fontsize=40,handletextpad=0.1,columnspacing=0.5)
    #plt.legend()
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)  # add space left
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
    if metric == 'mae':
        ax.set_xlim([0, 56])
    else:
        ax.set_xlim([0, 56])
    # # plt.xlim(0, 25)
    # plt.show()
    plt.savefig(main_path_for_save_fig+'/'+metric+'_ECDF_pers_dashed.pdf', bbox_inches='tight')
    plt.savefig(main_path_for_save_fig + '/' + metric + '_ECDF_pers_dashed.png', bbox_inches='tight')
    plt.close()