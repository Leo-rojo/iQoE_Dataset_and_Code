import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from matplotlib import cm
import warnings
#os.chdir('/Figure 6 and 8 and Table 1')
warnings.filterwarnings("ignore")
colori=cm.get_cmap('tab10').colors

path_iQoE='users'
maes=[]
rmses=[]
scores_more_users=[]
videos_more_users=[]
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

moses=np.mean(np.array(scores_more_users),axis=0)
#save moses in excel as vertical column
import pandas as pd
df = pd.DataFrame(moses)
df.to_excel('moses.xlsx', index=False, header=False)


collect_sumbit=np.load('features_qoes_train/feat_bit.npy')
collect_sumpsnr=np.load('features_qoes_train/feat_psnr.npy')
collect_sumssim=np.load('features_qoes_train/feat_ssim.npy')
collect_sumvmaf=np.load('features_qoes_train/feat_vmaf.npy')
collect_logbit=np.load('features_qoes_train/feat_logbit.npy')
collect_FTW=np.load('features_qoes_train/feat_ftw.npy')
collect_SDNdash=np.load('features_qoes_train/feat_sdn.npy')
collect_videoAtlas=np.load('features_qoes_train/feat_va.npy')

############################################################calculate personalized parameters#########################################################
collect_all_features=[collect_sumbit,collect_logbit,collect_sumpsnr,collect_sumssim,collect_sumvmaf,np.array(collect_FTW),collect_SDNdash,collect_videoAtlas]
users_scores=moses
l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
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

collect_all = []
for idx_i,i in enumerate(l):
    collect_temp = []
    all_features = collect_all_features[idx_i]

    if i == 'FTW':
        collect_temp.append(fit_nonlinear((all_features[:, 0], all_features[:, 1]), users_scores))
    elif i == 'videoAtlas':
        pickle.dump(fit_supreg(all_features, users_scores), open('./videoAtlas_mos.pkl', 'wb'))
        collect_temp.append('videoAtlas_user_' + identifier + '.pkl')
    else:
        collect_temp.append(fit_linear(all_features, users_scores))

    collect_all.append(collect_temp)
    #print(i)
np.save('params_mos',collect_all)

print('-----------train_done-----------')

#####################test##############################

maes=[]
rmses=[]
all_scores_all_models_all_users=[]#each cell user which has 8 models
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
        y_test = [int(i) for i in list(d_test.values())]

        idx_col_test = np.load(path_iQoE + '/original_database/idx_col_test.npy')

        collect_sumbit = np.load('features_qoes_test/feat_bit.npy')
        collect_sumpsnr = np.load('features_qoes_test/feat_psnr.npy')
        collect_sumssim = np.load('features_qoes_test/feat_ssim.npy')
        collect_sumvmaf = np.load('features_qoes_test/feat_vmaf.npy')
        collect_logbit = np.load('features_qoes_test/feat_logbit.npy')
        collect_FTW = np.load('features_qoes_test/feat_ftw.npy')
        collect_SDNdash = np.load('features_qoes_test/feat_sdn.npy')
        collect_videoAtlas = np.load('features_qoes_test/feat_va.npy')

        #calcola QoE of different model
        all_scores=[]
        temp_score_bits = []
        temp_score_logbits = []
        temp_score_psnr = []
        temp_score_ssim = []
        temp_score_vmaf = []
        temp_score_FTW = []
        temp_score_SDNdash = []
        temp_score_videoAtlas = []
        models = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
        params=np.load('params_mos.npy',allow_pickle=True)
        for kind_of_models in models:
            if kind_of_models == 'bit':  # [s_bit,s_dif_bit,s_psnr,s_dif_psnr,s_ssim,s_dif_ssim,s_vmaf,s_dif_vmaf,s_bit_log,s_dif_bit_log,ave_st_FTW,nr_stall_FTW,s_reb]
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[0][0], collect_sumbit[exp])  # here should go the non linear mapping eventually for real context.
                    temp_score_bits.append(score)
            elif kind_of_models == 'logbit':
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[1][0], collect_logbit[exp])
                    temp_score_logbits.append(score)
            elif kind_of_models == 'psnr':
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[2][0], collect_sumpsnr[exp])
                    temp_score_psnr.append(score)
            elif kind_of_models == 'ssim':
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[3][0], collect_sumssim[exp])
                    temp_score_ssim.append(score)
            elif kind_of_models == 'vmaf':
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[4][0], collect_sumvmaf[exp])
                    temp_score_vmaf.append(score)
            elif kind_of_models == 'FTW':
                for exp in range(len(idx_col_test)):
                    a, b, c, d = params[5][0]
                    x1, x2 = collect_FTW[exp]
                    score = a * np.exp(-(b * x1 + c) * x2) + d
                    temp_score_FTW.append(score)
            elif kind_of_models == 'SDNdash':
                for exp in range(len(idx_col_test)):
                    score = np.dot(params[6][0], collect_SDNdash[exp])
                    temp_score_SDNdash.append(score)
            elif kind_of_models == 'videoAtlas':
                with open('videoAtlas_mos.pkl','rb') as handle:
                    pickled_atlas = pickle.load(handle)
                videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
                temp_score_videoAtlas = videoAtlasregressor.predict(collect_videoAtlas)

        all_scores = [temp_score_bits, temp_score_logbits, temp_score_psnr, temp_score_ssim, temp_score_vmaf, temp_score_FTW, temp_score_SDNdash, temp_score_videoAtlas]
        all_scores_all_models_all_users.append(all_scores)

        #calculate metrics
        maes_u=[]
        rmses_u=[]
        # y_test=users_scores = [int(i) for i in list(d_test.values())]
        # scores iQoE
        for i in range(8):
            maes_u.append(mean_absolute_error(y_test, all_scores[i]))
            rmses_u.append(sqrt(mean_squared_error(y_test, all_scores[i])))
        maes.append(maes_u) #each cell of the array is a user and each cell of the user is a particular qoe model
        rmses.append(rmses_u)

#if does not exist create result_collection folder
if not os.path.exists('results_collection'):
    os.makedirs('results_collection')
np.save('results_collection/group_mae_each_user.npy',maes)
np.save('results_collection/group_rmse_each_user.npy',rmses)
print('-----------test_done-----------')








