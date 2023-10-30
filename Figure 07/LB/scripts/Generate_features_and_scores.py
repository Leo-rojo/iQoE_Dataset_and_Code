import numpy as np
import os
#insert path to your folder Figure 07
nr_c = 4
path_iQoE='../input_data/users'
maes=[]
rmses=[]
scores_more_users=[]
videos_more_users=[]
identifiers_order=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier

        ##train data
        # save dictonary idx_original-score
        a=[]
        a2=[]
        d_train = {}
        #exp_orig_train_t = []
        #scaled_exp_orig_train_t = []
        #take first 20 of train
        with open(user_folder + '/Scores_' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_train[int(key)] = val
                if len(d_train)==10:
                    break
        a.append([int(i) for i in list(d_train.values())])
        a2.append([int(i) for i in list(d_train.keys())])

        ###baseline data
        b=[]
        b2=[]
        d_baselines={}
        #exp_orig_train_b = []
        #scaled_exp_orig_train_b = []
        with open(user_folder + '/Scores_baseline' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_baselines[int(key)] = val
        b.append([int(i) for i in list(d_baselines.values())])
        b2.append([int(i) for i in list(d_baselines.keys())])
        scores_more_users.append(a[0]+b[0])
        videos_more_users.append(a2[0]+b2[0])
        identifiers_order.append(identifier)
np.save('../output_data/identifiers_order',identifiers_order)
np.save('../output_data/baselines_scores',scores_more_users)
idx_col_train = np.load(path_iQoE+'/original_database/idx_col_train.npy')
exp_orig = np.load(path_iQoE+'/original_database/synth_exp_train.npy')
scaled_exp_orig = np.load(path_iQoE+'/original_database/X_train_scaled.npy')

exp_orig_train = []
scaled_exp_orig_train = []
orig_orig=[]
merge_dictionary = {**d_train, **d_baselines}
for x in merge_dictionary.keys():  # array of original idxs
    idx_standard= np.where(idx_col_train == x)
    exp_orig_train.append(exp_orig[idx_standard])
    scaled_exp_orig_train.append(scaled_exp_orig[idx_standard])

##general bit
#take features
# bitrate_features
# collect features my experience
collect_sumbit = []
collect_sumpsnr = []
collect_sumssim = []
collect_sumvmaf = []
collect_logbit = []
collect_FTW = []
collect_SDNdash = []
collect_videoAtlas = []
#min training bitrate
bit = []
for exp in exp_orig:
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
min_bit=np.min(bit)

for exp in exp_orig_train:
    exp=exp[0]
    bit = []
    logbit = []
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_bit)
        logbit.append(bit_log)
    # sumbit
    s_bit = np.array(bit).sum()
    # sumlogbit
    l_bit = np.array(logbit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    # ave of all reb
    s_reb_ave = np.array(reb).mean()
    # nr of stall
    nr_stall = np.count_nonzero(reb)
    # duration stall+normal
    tot_dur_plus_reb = nr_c * 2 + s_reb

    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10 - 1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    # ssim
    ssim = []
    for i in range(8, (8 + nr_c * 10 - 1), 10):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    # vmaf
    vmaf = []
    for i in range(9, (9 + nr_c * 10 - 1), 10):
        vmaf.append(float(exp[i]))
    # sum
    s_vmaf = np.array(vmaf).sum()
    # ave
    s_vmaf_ave = np.array(vmaf).mean()

    # is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(6, (6 + nr_c * 10 - 1), 10):
        isbest.append(float(exp[i]))

    is_best = np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 2
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([4 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    # collection
    collect_sumbit.append([s_bit, s_reb, s_dif_bit])
    collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
    collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
    collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

    collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
    collect_FTW.append([s_reb_ave, nr_stall])
    collect_SDNdash.append([s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
    collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])

if not os.path.exists('../output_data/../output_data/features_qoes_train'):
    os.makedirs('../output_data/../output_data/features_qoes_train')
np.save('../output_data/features_qoes_train/feat_vmaf', collect_sumvmaf)
np.save('../output_data/features_qoes_train/feat_va', collect_videoAtlas)
np.save('../output_data/features_qoes_train/feat_sdn', collect_SDNdash)
np.save('../output_data/features_qoes_train/feat_ftw', collect_FTW)
np.save('../output_data/features_qoes_train/feat_psnr', collect_sumpsnr)
np.save('../output_data/features_qoes_train/feat_ssim', collect_sumssim)
np.save('../output_data/features_qoes_train/feat_bit', collect_sumbit)
np.save('../output_data/features_qoes_train/feat_logbit', collect_logbit)



########test#############
test_scores=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier
        #print(identifier)

        ##test data
        #save dictonary idx_original-score
        d_test = {}
        exp_orig_test=[]
        scaled_exp_orig_test=[]
        with open(user_folder+'/Scores_test_'+identifier+'.txt') as f:
            for line in f:
               val = line.split()[-1]
               nextline=next(f)
               key = nextline.split()[-1]
               d_test[int(key)] = val
        test_scores.append([int(i) for i in list(d_test.values())])
np.save('../output_data/test_scores',test_scores)
idx_col_test = np.load(path_iQoE + '/original_database/idx_col_test.npy')
exp_orig = np.load(path_iQoE+'/original_database/synth_exp_test.npy')
scaled_exp_orig = np.load(path_iQoE+'/original_database/X_test_scaled.npy')
for x in d_test.keys():  # array of original idxs
    idx_standard = np.where(idx_col_test == x)
    exp_orig_test.append(exp_orig[idx_standard])
    scaled_exp_orig_test.append(scaled_exp_orig[idx_standard])

collect_sumbit = []
collect_sumpsnr = []
collect_sumssim = []
collect_sumvmaf = []
collect_logbit = []
collect_FTW = []
collect_SDNdash = []
collect_videoAtlas = []
#min training bitrate
bit = []
for exp in exp_orig:
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
min_bit=np.min(bit)

for exp in exp_orig_test:
    exp=exp[0]
    bit = []
    logbit = []
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_bit)
        logbit.append(bit_log)
    # sumbit
    s_bit = np.array(bit).sum()
    # sumlogbit
    l_bit = np.array(logbit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):  # start from 11 and not 1 exactly for the reason explained before
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    # ave of all reb
    s_reb_ave = np.array(reb).mean()
    # nr of stall
    nr_stall = np.count_nonzero(reb)
    # duration stall+normal
    tot_dur_plus_reb = nr_c * 2 + s_reb

    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10 - 1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    # ssim
    ssim = []
    for i in range(8, (8 + nr_c * 10 - 1), 10):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    # vmaf
    vmaf = []
    for i in range(9, (9 + nr_c * 10 - 1), 10):
        vmaf.append(float(exp[i]))
    # sum
    s_vmaf = np.array(vmaf).sum()
    # ave
    s_vmaf_ave = np.array(vmaf).mean()

    # is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(6, (6 + nr_c * 10 - 1), 10):
        isbest.append(float(exp[i]))

    is_best = np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 2
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([4 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    # collection
    collect_sumbit.append([s_bit, s_reb, s_dif_bit])
    collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
    collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
    collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

    collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
    collect_FTW.append([s_reb_ave, nr_stall])
    collect_SDNdash.append([s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
    collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])

if not os.path.exists('../output_data/features_qoes_test'):
    os.makedirs('../output_data/features_qoes_test')
np.save('../output_data/features_qoes_test/feat_vmaf', collect_sumvmaf)
np.save('../output_data/features_qoes_test/feat_va', collect_videoAtlas)
np.save('../output_data/features_qoes_test/feat_sdn', collect_SDNdash)
np.save('../output_data/features_qoes_test/feat_ftw', collect_FTW)
np.save('../output_data/features_qoes_test/feat_logbit', collect_logbit)
np.save('../output_data/features_qoes_test/feat_bit', collect_sumbit)
np.save('../output_data/features_qoes_test/feat_psnr', collect_sumpsnr)
np.save('../output_data/features_qoes_test/feat_ssim', collect_sumssim)
