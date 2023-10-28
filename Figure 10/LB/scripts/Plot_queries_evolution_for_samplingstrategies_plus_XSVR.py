import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from matplotlib import cm
import os

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

n_queries=250
regr_choosen='SVR'


for metric in ['mae','rmse']:
    save_five_ss = []
    save_five_ss_stdv = []
    for initialq in ['no_R','R']:

        if initialq=='no_R':
            span=range(4)
        else:
            span=[0,4] #['RS+SVR', 'RCU+SVR', 'RGS+SVR', 'RQBC+SVR', 'iQoE']
        for ss in span:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+initialq
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_five_ss_stdv.append(np.std(save_ave_across_users_for_shuffle,axis=0))
            print(main_path)
    main_path_for_save_fig = 'Plot_continuous_metrics_' + regr_choosen+'ynR'
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    np.save(main_path_for_save_fig+'/'+metric+'values',save_five_ss)
    #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_five_ss_stdv)

    fig = plt.figure(figsize=(20, 10),dpi=100)
    leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR']+['RS+SVR','iQoE'] #iQoE=igs+SVR
    colors=[colori[1],colori[2],colori[4],colori[5]]+[colori[0],'r']
    stile=[':','--','-.','-.']+['--','-']
    conta = 0
    for kind_strategy in leg:
        kind_strategy_idx=leg.index(kind_strategy)
        users_meanmean=save_five_ss[conta][10:]
        users_meanster=save_five_ss_stdv[conta][10:]
        f = interp1d(range(n_queries+1-10), users_meanmean)
        plt.plot(range(n_queries+1-10), f(range(n_queries+1-10)), stile[conta], linewidth='7', label=leg[conta],color=colors[conta])
        conta += 1
    #plt.grid()
    plt.xlabel("Number of experiences", fontdict=font_axes_titles)
    plt.xticks([0, 50-10, 100-10, 150-10, 200-10, 250-10], ['10', '50', '100', '150', '200', '250'])
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)

    plt.yticks(np.arange(0,15,2))
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24)
    ax.set_ylim([1.3, 16])
    plt.savefig(main_path_for_save_fig + '/' + metric+'comb_10.pdf',bbox_inches='tight')
    plt.close()

for metric in ['mae','rmse']:
    save_five_ss = []
    save_five_ss_stdv = []
    for initialq in ['no_R','R']:

        if initialq == 'no_R':
            span = range(4)
        else:
            span = [0, 4]  # ['RS+SVR', 'RCU+SVR', 'RGS+SVR', 'RQBC+SVR', 'iQoE']
        for ss in span:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+initialq
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_five_ss_stdv.append(np.std(save_ave_across_users_for_shuffle,axis=0))
    main_path_for_save_fig = 'Plot_continuous_metrics_' + regr_choosen+'ynR'
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    np.save(main_path_for_save_fig+'/'+metric+'values',save_five_ss)
    #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_five_ss_stdv)

    fig = plt.figure(figsize=(20, 10),dpi=100)
    leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR'] + ['RS+SVR', 'iQoE']  # iQoE=igs+SVR
    colors = [colori[1], colori[2], colori[4], colori[5]] + [colori[0], 'r']
    stile = [':', '--', '-.', '-.'] + ['--', '-']
    conta = 0
    for kind_strategy in leg:
        kind_strategy_idx=leg.index(kind_strategy)
        users_meanmean=save_five_ss[conta]
        users_meanster=save_five_ss_stdv[conta]
        f = interp1d(range(n_queries+1), users_meanmean)
        plt.plot(range(n_queries+1), f(range(n_queries+1)), stile[conta], linewidth='7', label=leg[conta],color=colors[conta])
        conta += 1
    #plt.grid()
    plt.xlabel("Number of experiences", fontdict=font_axes_titles)
    plt.xticks([0, 50, 100, 150, 200, 250], ['0', '50', '100', '150', '200', '250'])
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)

    plt.yticks(np.arange(0,15,2))
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24)
    ax.set_ylim([1.3, 16])
    plt.savefig(main_path_for_save_fig + '/' + metric+'_comb_all.pdf',bbox_inches='tight')
    plt.close()





