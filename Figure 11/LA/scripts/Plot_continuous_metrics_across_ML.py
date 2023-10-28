from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm # recommended import according to the docs
from matplotlib.patches import Patch
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

#for ML model
AL_stra='RiGS'
n_queries=250
ss=0 #I have runned experiments only with rigs

#after10
for metric in ['mae','rmse']:
        save_three_reg=[]
        save_three_reg_std=[]
        for regr_choosen in ['SVR','XGboost','RF','GP']:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path='../input_data/'+regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+'rigs'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users = save_ave_across_users
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_three_reg.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_three_reg_std.append(np.std(save_ave_across_users_for_shuffle,axis=0))

        main_path_for_save_fig = '../'
        np.save(main_path_for_save_fig + '/' + metric + 'values', save_three_reg)
        #np.save(main_path_for_save_fig + '/' + metric + 'values'+flag_insca, save_three_reg_std)

        fig = plt.figure(figsize=(20, 10),dpi=100)
        leg = ['iQoE','iGS+XGB','iGS+RF','iGS+GP']
        conta = 0
        stile = ['-', '--', '-.', ':']
        col=['r',colori[2],colori[4],colori[0]]
        for regr in ['iQoE','XGB+iGS','RF+iGS','GP+iGS']:
            users_meanmean=save_three_reg[conta][10:]
            users_meanster=save_three_reg_std[conta][10:]
            #data1 = plt.scatter(range(n_queries+1-20), users_meanmean, marker='.',color=col[conta])
            #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
            f = interp1d(range(n_queries+1-10), users_meanmean)
            plt.plot(range(n_queries+1-10), f(range(n_queries+1-10)), stile[conta], linewidth='7',color=col[conta])
            conta += 1
        #plt.grid()
        plt.xlabel("Number of experiences", fontdict=font_axes_titles)

        plt.xticks([0, 50-10, 100-10, 150-10, 200-10, 250-10], ['10', '50', '100', '150', '200', '250'])
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)
        #plt.yticks(range(0, 20, 2))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.yticks(np.arange(0, 17, 2))
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24)
        ax.set_ylim([1.3, 17])
        #lege = [leg[-1], leg[1], leg[2], leg[3], leg[0]]
        #colorsi = [col[-1], col[1], col[2], col[3], col[0]]
        handles = [Patch(facecolor=color, label=label) for label, color in zip(leg, col)]
        #plt.legend(ncol=2,frameon=False, handles=handles, handlelength=2., handleheight=0.7, fontsize=40,bbox_to_anchor=(0.03, 0.07, 1, 1),handletextpad=0.1,columnspacing=0.5)
        #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
        #plt.legend(ncol=3,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
        #plt.show()
        #plt.savefig(main_path_for_save_fig + '/' + metric+'ml_10.pdf',bbox_inches='tight',)
        plt.close()

#all
for metric in ['mae','rmse']:
        save_three_reg=[]
        save_three_reg_std=[]
        for regr_choosen in ['SVR','XGboost','RF','GP']:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path='../input_data/'+regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+'rigs'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users = save_ave_across_users
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_three_reg.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_three_reg_std.append(np.std(save_ave_across_users_for_shuffle,axis=0))

        main_path_for_save_fig = '../'
        np.save(main_path_for_save_fig + '/' + metric + 'values', save_three_reg)
        #np.save(main_path_for_save_fig + '/' + metric + 'values'+flag_insca, save_three_reg_std)

        fig = plt.figure(figsize=(20, 10),dpi=100)
        leg = ['iQoE','iGS+XGB','iGS+RF','iGS+GP']
        conta = 0
        stile = ['-', '--', '-.', ':']
        col=['r',colori[2],colori[4],colori[0]]
        for regr in ['iQoE','XGB+iGS','RF+iGS','GP+iGS']:
            users_meanmean=save_three_reg[conta]
            users_meanster=save_three_reg_std[conta]
            #data1 = plt.scatter(range(n_queries+1-20), users_meanmean, marker='.',color=col[conta])
            #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
            f = interp1d(range(n_queries+1), users_meanmean)
            plt.plot(range(n_queries+1), f(range(n_queries+1)), stile[conta], linewidth='7',color=col[conta])
            conta += 1
        #plt.grid()
        plt.xlabel("Number of experiences", fontdict=font_axes_titles)

        plt.xticks([0, 50, 100, 150, 200, 250], ['0', '50', '100', '150', '200', '250'])
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)
        #plt.yticks(range(0, 20, 2))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.yticks(np.arange(0, 17, 2))
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24)
        ax.set_ylim([1.3, 17])
        #lege = [leg[-1], leg[1], leg[2], leg[3], leg[0]]
        #colorsi = [col[-1], col[1], col[2], col[3], col[0]]
        handles = [Patch(facecolor=color, label=label) for label, color in zip(leg, col)]
        #plt.legend(ncol=2,frameon=False, handles=handles, handlelength=2., handleheight=0.7, fontsize=40,bbox_to_anchor=(0.03, 0.07, 1, 1),handletextpad=0.1,columnspacing=0.5)
        #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
        #plt.legend(ncol=3,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
        #plt.show()
        plt.savefig(main_path_for_save_fig + '/' + metric+'ml_all.pdf',bbox_inches='tight',)
        plt.close()








