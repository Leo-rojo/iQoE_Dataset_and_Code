import statsmodels.api as sm # recommended import according to the docs
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

n_queries=250
leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR'] + ['RS+SVR', 'iQoE']  # iQoE=igs+SVR
colors = [colori[1], colori[2], colori[4], colori[5]] + [colori[0], 'r']
stile = [':', '--', '-.', '-.'] + ['--', '-']
regr_choosen='SVR'

for metric in ['mae','rmse']:
    save_five_ss = []
    for initialq in ['no_R', 'R']:

        if initialq == 'no_R':
            span = range(4)
        else:
            span = [0, 4]  # ['RS+SVR', 'RCU+SVR', 'RGS+SVR', 'RQBC+SVR', 'iQoE']
        for ss in span:
            save_ave_across_users_for_shuffle = []
            for shuffle in [13, 34, 42, 70, 104]:
                save_ave_across_users = []
                for u in range(32):
                    for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                        main_path = '../input_data/'+regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+initialq
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                            ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(save_ave_across_users)
            save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
    #np.save('values'+metric+totuser,save_five_ss)


    fix_query=49

    main_path_for_save_fig = '../'
    #np.save(main_path_for_save_fig + '/' + metric + 'values_ave', final_ave)
    #np.save(main_path_for_save_fig + '/' + metric + 'values_std', final_std)
    save_distributions=[]
    fig = plt.figure(figsize=(20, 10), dpi=100)
    conta = 0

    for ss in range(6):

        final_ave_elab=[save_five_ss[ss][i][fix_query] for i in range(256)]
        ecdf = sm.distributions.ECDF(final_ave_elab)
        save_distributions.append(ecdf)
        plt.step(ecdf.x, ecdf.y, label=leg[ss], linewidth=7.0, color=colors[ss], linestyle=stile[conta])
        conta+=1
    np.save(main_path_for_save_fig+'/'+'values_ecdf'+metric,save_distributions)

    plt.xlabel(metric.upper(), fontdict=font_axes_titles)
    plt.ylabel('% of raters', fontdict=font_axes_titles)
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)  # add space left
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
    ax.set_xlim([0, 20])
    # plt.xlim(0, 25)
    # plt.show()
    plt.savefig(main_path_for_save_fig+'/'+metric+'_comb_ECDF.pdf', bbox_inches='tight')
    plt.close()







