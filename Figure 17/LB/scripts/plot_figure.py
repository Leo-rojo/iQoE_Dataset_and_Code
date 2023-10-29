import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
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

regr_choosen='SVRigs'

for metric in ['mae','rmse']:#,'lcc','srcc']:
    save_ts=[]
    save_tsstdv=[]
    for t_s in [0.99,0.97,0.95,0.93,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        save_ave_across_users_for_shuffle=[]
        for shuffle in [13,34,42,70,104]:
            save_ave_across_users=[]
            for u in range(32):
                for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                    if t_s in [0.99,0.97,0.95,0.93]:
                        n_queries=int((round(1-t_s,2)*1000)-1)
                        spli=str(round(1 - t_s, 2))
                    else:
                        n_queries = 99
                        spli=str(t_s)
                    main_path='../output_data/'+regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_'+spli
                    user_data = np.load('./' + main_path + '/' + QoE_model + '/user_' + str(
                        u) +'/ts_'+ spli + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')#[ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
        only100=np.mean(save_ave_across_users_for_shuffle, axis=0)
        only100std=np.std(save_ave_across_users_for_shuffle, axis=0)
        save_ts.append(only100[0])
        save_tsstdv.append(only100std[0])

    main_path_for_save_fig = '../'
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    #np.save(main_path_for_save_fig+'/'+metric+'values',save_ts)
    #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_tsstdv)

    if metric=='lcc':
        metric='plcc'
    if metric=='srcc':
        metric='srocc'
    # fig = plt.figure(figsize=(20, 10),dpi=100)
    leg = ['0.01','0.03','0.05','0.07','0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8','0.9']

    #barre
    nr_query_plot=49
    conta=0
    users_meanmean_50=[]
    users_meanmean_75 = []
    users_meanmean_100 = []
    users_meanmeanstd=[]
    fig = plt.figure(figsize=(20, 10), dpi=100)
    for kind_strategy in leg:
        kind_strategy_idx = leg.index(kind_strategy)
        if kind_strategy in ['0.01', '0.03', '0.05', '0.07']:
            users_meanmean_50.append(save_ts[conta][-1])
            users_meanmean_75.append(save_ts[conta][-1])
            users_meanmean_100.append(save_ts[conta][-1])
        else:
            users_meanmean_50.append(save_ts[conta][49])
            users_meanmean_75.append(save_ts[conta][74])
            users_meanmean_100.append(save_ts[conta][99])
        #users_meanmeanstd.append(save_tsstdv[conta][nr_query_plot])
        conta += 1
    l = [0.01, 0.03, 0.05, 0.07] + (np.arange(0, 0.9, 0.1) + 0.1).tolist()
    #np.save(main_path_for_save_fig + '/' + metric + 'values_50', users_meanmean_50)
    #np.save(main_path_for_save_fig + '/' + metric + 'values_75', users_meanmean_75)
    #np.save(main_path_for_save_fig + '/' + metric + 'values_100', users_meanmean_100)
    plt.plot(l,users_meanmean_50, '-', linewidth='7', label='after 50 SAs',color='r')  # , label=leg[conta], color=colors[conta])'r',colori[2],colori[4]
    plt.plot(l,users_meanmean_75, '--', linewidth='7', label='after 100 SAs',color=colori[2], )  # , label=leg[conta], color=colors[conta])
    plt.plot(l,users_meanmean_100, ':', linewidth='7', label='after 150 SAs',color=colori[0])  # , label=leg[conta], color=colors[conta])
    #plt.grid()
    plt.xlabel("Training-set share", fontdict=font_axes_titles)

    plt.xticks([0]+l, ['0','','','','','0.1', '0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])  # , ['1', '50', '100', '150', '200', '250+'])
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 13, 2))
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.margins(0.013, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24)
    xticks = ax.xaxis.get_major_ticks()
    for i in range(1,5):
        xticks[i].set_visible(False)
    ax.set_ylim([2,12])
    plt.savefig(main_path_for_save_fig + '/' + metric +  '_line'+'.pdf', bbox_inches='tight')
    plt.close()





