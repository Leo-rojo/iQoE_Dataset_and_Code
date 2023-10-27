import pickle
import warnings
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os
#insert path to your folder Figure 9
warnings.filterwarnings("ignore")
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
def sigmoid(x, k, x0):
    return (99.0 / (1 + np.exp(-k * (x - x0)))) + 1
def findmaxlistinlist(lista):
    ar=[]
    for i in lista:
        a=[k for k in i]
        ar.append(sum(a))
    return np.nanargmax(ar)

all_features=[]
#collect features and scores of real users
path_iQoE= 'users'
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        print(identifier)
        user_folder = path_iQoE+'/user_' + identifier

        ##test
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
        idx_col_test = np.load(path_iQoE+'/original_database/idx_col_test.npy')
        y_test = [int(i) for i in list(d_test.values())]


        exp_orig = np.load(path_iQoE+'/original_database/synth_exp_test.npy')
        scaled_exp_orig = np.load(path_iQoE+'/original_database/X_test_scaled.npy')
        for x in d_test.keys():  # array of original idxs
            idx_standard = np.where(idx_col_test == x)
            exp_orig_test.append(exp_orig[idx_standard])
            scaled_exp_orig_test.append(scaled_exp_orig[idx_standard])


        ##train
        # save dictonary idx_original-score
        d_train = {}
        exp_orig_train = []
        scaled_exp_orig_train = []
        with open(user_folder + '/Scores_' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_train[int(key)] = val

        idx_col_train = np.load(path_iQoE+'/original_database/idx_col_train.npy')
        exp_orig = np.load(path_iQoE+'/original_database/synth_exp_train.npy')
        scaled_exp_orig = np.load(path_iQoE+'/original_database/X_train_scaled.npy')
        for x in d_train.keys():  # array of original idxs
            idx_standard = np.where(idx_col_train == x)
            exp_orig_train.append(exp_orig[idx_standard])
            scaled_exp_orig_train.append(scaled_exp_orig[idx_standard])
        y_train = [int(i) for i in list(d_train.values())]


        #baselines
        ##train data
        # save dictonary idx_original-score
        d_baselines = {}
        with open(user_folder + '/Scores_baseline' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_baselines[int(key)] = val

        idx_col_train = np.load(path_iQoE + '/original_database/idx_col_train.npy')
        exp_orig = np.load(path_iQoE + '/original_database/synth_exp_train.npy')
        scaled_exp_orig = np.load(path_iQoE + '/original_database/X_train_scaled.npy')

        exp_orig_train_baseline = []
        scaled_exp_orig_train_baseline = []
        for x in d_baselines.keys():  # array of original idxs
            idx_standard = np.where(idx_col_train == x)
            exp_orig_train_baseline.append(exp_orig[idx_standard])
            scaled_exp_orig_train_baseline.append(scaled_exp_orig[idx_standard])
        y_baseline = [int(i) for i in list(d_baselines.values())]

        #remove last column
        exp_orig_train_baseline_nolast=[i[0][:-1] for i in exp_orig_train_baseline]
        exp_orig_train_nolast = [i[0][:-1] for i in exp_orig_train]
        exp_orig_test_nolast = [i[0][:-1] for i in exp_orig_test]


        #train
        import pandas as pd
        rft_train=[list(i) for i in exp_orig_train_nolast]
        for c,i in enumerate(y_train):
            rft_train[c].append(i)
        #baseline
        rft_baseline = [list(i) for i in exp_orig_train_baseline_nolast]
        for c, i in enumerate(y_baseline):
            rft_baseline[c].append(i)
        #test
        rft_test = [list(i) for i in exp_orig_test_nolast]
        for c, i in enumerate(y_test):
            rft_test[c].append(i)

        rft=rft_train+rft_baseline+rft_test

        all_features.append(rft)


###calcuate features for real exp for each real users and scores
#scores
us_scores_vectors=[]
for i in all_features:
    each_user=[]
    for k in range(len(i)):
        each_user.append(i[k][-1])
    us_scores_vectors.append(each_user)


nr_c=4
all_us_f=[]
for userfeatures in all_features:
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
    for exp in userfeatures:
        for i in range(2, (2 + nr_c * 10 - 1), 10):
            bit.append(float(exp[i]))
    min_bit=np.min(bit)

    for exp in userfeatures:
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
        tot_dur_plus_reb = nr_c * 4 + s_reb

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
                m += 4
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
        collect_SDNdash.append(
            [s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
        collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])
    all_us_f.append([collect_sumbit, collect_logbit, collect_sumpsnr, collect_sumssim, collect_sumvmaf, collect_FTW, collect_SDNdash,collect_videoAtlas])

#synthetic user collection
#give scores
models=['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']
models_folder='Fitted_models_without_logistic'
params_sigmoid=np.load('./save_param_sigmoids/params_sigmoid.npy')

#load synthetic models (which means parameters)
all_synthetic_users=[]
for u in range(32):
    synthetic_user_models=[]
    for model in models:
        if model=='videoAtlas':
            with open('./'+models_folder+'/organized_by_users/user_'+str(u)+'/model_videoAtlas.pkl', 'rb') as handle:
                synthetic_user_models.append(pickle.load(handle))
        else:
            synthetic_user_models.append(np.load('./'+models_folder+'/organized_by_users/user_'+str(u)+'/model_'+model+'.npy',allow_pickle=True))
    all_synthetic_users.append(synthetic_user_models)



#all_us_f[user][model] #model=0 and user=0 means that we have 120 group of 3 feat to make score the bitrate synthetic users


nr_of_exp=120
all_real_us_all_su_scores=[]
for real_user in range(34): #per ogni real user
    scores_of_su_for_ru=[]
    for synthetic_user in range(32): #per ogni synthetic user
        eight_models_for_each_su=all_synthetic_users[synthetic_user]
        for kind_of_models in models:
            if kind_of_models=='bit':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[0],all_us_f[real_user][0][exp])) #bitsuscore=np.dot(user_models[0],all_us_f[real_user][0][exp])
                b_sig, c_sig = params_sigmoid[synthetic_user][0]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='logbit':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[1],all_us_f[real_user][1][exp]))
                b_sig, c_sig = params_sigmoid[synthetic_user][1]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='psnr':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[2],all_us_f[real_user][2][exp]))
                b_sig, c_sig = params_sigmoid[synthetic_user][2]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='ssim':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[3],all_us_f[real_user][3][exp]))
                b_sig, c_sig = params_sigmoid[synthetic_user][3]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='vmaf':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[4],all_us_f[real_user][4][exp]))
                b_sig, c_sig = params_sigmoid[synthetic_user][4]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='FTW':
                scores=[]
                for exp in range(nr_of_exp):
                    a, b, c, d = eight_models_for_each_su[5]
                    x1, x2 = all_us_f[real_user][5][exp]
                    scores.append(a * np.exp(-(b * x1 + c) * x2) + d)
                b_sig, c_sig = params_sigmoid[synthetic_user][5]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
            elif kind_of_models=='SDNdash':
                scores=[]
                for exp in range(nr_of_exp):
                    scores.append(np.dot(eight_models_for_each_su[6],all_us_f[real_user][6][exp]))
                b_sig, c_sig = params_sigmoid[synthetic_user][6]
                scores_120_vector=sigmoid(scores, b_sig, c_sig)
                scores_of_su_for_ru.append(scores_120_vector)
    all_real_us_all_su_scores.append(scores_of_su_for_ru)

all_reals_with_all=[]
for real_user in range(34):
    real_with_all=[]
    for synthetic_user in range(224):
        score_real=us_scores_vectors[real_user]
        score_synth=all_real_us_all_su_scores[real_user][synthetic_user]
        #calculate person,spearman between score_real and score_synth
        corr_p, _ = pearsonr(score_real, score_synth)
        # calculate spearman correlation between user_scores
        corr_s, _ = spearmanr(score_real, score_synth)
        # calculate kendall correlation between user_scores
        real_with_all.append([corr_p, corr_s])
    all_reals_with_all.append(real_with_all)


pers_others=[]
spears_others=[]
for real_user in range(34):
    max_index=findmaxlistinlist(all_reals_with_all[real_user])
    #print(all_reals_with_all[real_user][max_index],max_index)
    for i,j in enumerate(all_reals_with_all[real_user][max_index]):
        if i == 0:
            pers_others.append(j)
        elif i == 1:
            spears_others.append(j)

#plot ECDF of pearson correlation, spearman correlation

lab=['Pearson', 'Spearman']
fig = plt.figure(figsize=(20, 10), dpi=100)
style=['-','--']
save_distributions=[]
for c,corr_metric in enumerate([pers_others,spears_others]):
    ecdf = sm.distributions.ECDF(corr_metric)
    save_distributions.append(ecdf)
    plt.step(ecdf.x, ecdf.y, label=lab[c], linewidth=7.0, color=['r','g'][c],linestyle=style[c])
np.save('ECDFvalues', save_distributions)

plt.xlabel('Correlation', fontdict=font_axes_titles)
plt.ylabel('% of raters', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','20','40','60','80','100'])
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
ax.set_xlim([0, 1.05])
#legend1 = ax.legend(loc='center',bbox_to_anchor=[0.50, 1.15],ncol=2, frameon=False,fontsize = 45)
plt.savefig('synthetic_user_validation.pdf', bbox_inches='tight')
plt.close()

#plot legend
import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)
#assing a random array of values to the variables
pearson=[1,2,3,4,5,6,7,8]
spearman=[1,2,3,4,5,6,7,8]
# create a figure for the data

conta = 0
style=['-','--']
col=['r','g']
names=['Pearson', 'Spearman']
for nr,i in enumerate([pearson,spearman]):
    #print(regr)
    plt.plot(i,style[nr], linewidth='7',color=col[nr],label=names[nr])
    conta += 1
ax = pylab.gca()
figLegend = pylab.figure(figsize = (20,10),dpi=100)
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=11,frameon=False,columnspacing=0.6,handletextpad=0.2,handlelength=1.45)
figLegend.savefig("correlation_leg.pdf",bbox_inches='tight')