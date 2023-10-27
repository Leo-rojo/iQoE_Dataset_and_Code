import numpy as np

n_queries=250
leg=['CU','GS','QBC','iGS']
foldforplot='.'
fix_q=49 #100

save_three_reg_igs=[]
save_three_reg_std_igs=[]
for metric in ['mae','rmse']:
    ss = 3  # igs
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle=[]
        for shuffle in [13,34,42,70,104]:
            save_ave_across_users=[]
            for u in range(32):
                for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                    main_path= regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+'no_R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
        save_three_reg_igs.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
        save_three_reg_std_igs.append(np.std(save_ave_across_users_for_shuffle,axis=0))
stringa= 'iGS'
for i,reg in enumerate(['XGboost','SVR','RF','GP','XGboost','SVR','RF','GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_igs[i][fix_q],1)) + '±' + str(round(save_three_reg_std_igs[i][fix_q],1))
stringa=stringa +'  \\\ '+'\\'+'hline'
print(stringa)
#######################################################
save_three_reg_qbc = []
save_three_reg_std_qbc = []
for metric in ['mae', 'rmse']:
    ss = 2  # qbc
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle = []
        for shuffle in [13, 34, 42, 70, 104]:
            save_ave_across_users = []
            for u in range(32):
                for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                    main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+ 'no_R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                        ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users, axis=0))
        save_three_reg_qbc.append(np.mean(save_ave_across_users_for_shuffle, axis=0))
        save_three_reg_std_qbc.append(np.std(save_ave_across_users_for_shuffle, axis=0))
stringa = 'QBC'
for i, reg in enumerate(['XGboost', 'SVR', 'RF', 'GP', 'XGboost', 'SVR', 'RF', 'GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_qbc[i][fix_q], 1)) + '±' + str(round(save_three_reg_std_qbc[i][fix_q], 1))
stringa = stringa + '  \\\ ' + '\\' + 'hline'
print(stringa)
###########################################################
save_three_reg_gs = []
save_three_reg_std_gs = []
for metric in ['mae', 'rmse']:
    ss = 1  # gs
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle = []
        for shuffle in [13, 34, 42, 70, 104]:
            save_ave_across_users = []
            for u in range(32):
                for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                    main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1' + 'no_R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                        ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users, axis=0))
        save_three_reg_gs.append(np.mean(save_ave_across_users_for_shuffle, axis=0))
        save_three_reg_std_gs.append(np.std(save_ave_across_users_for_shuffle, axis=0))
stringa = 'GS'
for i, reg in enumerate(['XGboost', 'SVR', 'RF', 'GP', 'XGboost', 'SVR', 'RF', 'GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_gs[i][fix_q], 1)) + '±' + str(round(save_three_reg_std_gs[i][fix_q], 1))
stringa = stringa + '  \\\ ' + '\\' + 'hline'
print(stringa)
##########################################################
save_three_reg_cu = []
save_three_reg_std_cu = []
for metric in ['mae', 'rmse']:
    ss = 0  # cu
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle = []
        for shuffle in [13, 34, 42, 70, 104]:
            save_ave_across_users = []
            for u in range(32):
                for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                    main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1' + 'no_R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                        ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users, axis=0))
        save_three_reg_cu.append(np.mean(save_ave_across_users_for_shuffle, axis=0))
        save_three_reg_std_cu.append(np.std(save_ave_across_users_for_shuffle, axis=0))
stringa = 'UC'
for i, reg in enumerate(['XGboost', 'SVR', 'RF', 'GP', 'XGboost', 'SVR', 'RF', 'GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_cu[i][fix_q], 1)) + '±' + str(round(save_three_reg_std_cu[i][fix_q], 1))
stringa = stringa + '  \\\ ' + '\\' + 'hline'
print(stringa)
##########################################################
save_three_reg_random = []
save_three_reg_std_random = []
for metric in ['mae', 'rmse']:
    ss = 0  # random
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle = []
        for shuffle in [13, 34, 42, 70, 104]:
            save_ave_across_users = []
            for u in range(32):
                for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                    main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1' + 'R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                        ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users, axis=0))
        save_three_reg_random.append(np.mean(save_ave_across_users_for_shuffle, axis=0))
        save_three_reg_std_random.append(np.std(save_ave_across_users_for_shuffle, axis=0))
stringa = 'RS'
for i, reg in enumerate(['XGboost', 'SVR', 'RF', 'GP', 'XGboost', 'SVR', 'RF', 'GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_random[i][fix_q], 1)) + '±' + str(round(save_three_reg_std_random[i][fix_q], 1))
stringa = stringa + '  \\\ ' + '\\' + 'hline'
print(stringa)
##########################################################
save_three_reg_igs=[]
save_three_reg_std_igs=[]
for metric in ['mae','rmse']:
    ss = 4  # igs
    for regr_choosen in ['XGboost','SVR','RF','GP']:
        save_ave_across_users_for_shuffle=[]
        for shuffle in [13,34,42,70,104]:
            save_ave_across_users=[]
            for u in range(32):
                for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                    main_path= regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'+'R'
                    user_data = np.load(foldforplot + '/' + main_path + '/' + QoE_model + '/user_' + str(
                        u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
        save_three_reg_igs.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
        save_three_reg_std_igs.append(np.std(save_ave_across_users_for_shuffle,axis=0))
stringa= 'RiGS'
for i,reg in enumerate(['XGboost','SVR','RF','GP','XGboost','SVR','RF','GP']):
    stringa = stringa + ' & ' + str(round(save_three_reg_igs[i][fix_q],1)) + '±' + str(round(save_three_reg_std_igs[i][fix_q],1))
stringa=stringa +'  \\\ '+'\\'+'hline'
print(stringa)









