from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import xgboost as xgb
from modAL.models import ActiveLearner
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn import linear_model
from scipy.optimize import curve_fit
import sklearn
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

#be in folder fig4


#take all individual user scores saved previously
users_scores=np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')
users_scores=users_scores.reshape(256,1000)

#take mos scores
mosarray=np.mean(users_scores,axis=0) # da splittare in 70-30

#function to fit video_atlas
def fit_supreg(all_features,mosscore):
    data = np.array(all_features)
    target = np.array(mosscore)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_
#function to fit linear models
def fit_linear(all_features,mosscore):
    # multi-linear model fitting
    X = all_features
    y = mosscore

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y)

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]
#function to fit non linear models
def fit_nonlinear(all_features,mosscore):
    def fun(data,a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, mosscore, maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d
#function for iGS
def random_greedy_sampling_input_output(regressor, X): #it is iGS
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(regressor.X_training, X)
    dist_y_matrix = pairwise_distances(
        regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]

#load saved IFs for each model and for each generated experience
features_folder='features_generated_experiences'
#load all bitrate features
all_features_bit = np.load('./'+features_folder+'/feat_bit_for_synth_exp.npy')
#load all bitrate features
all_features_psnr = np.load('./'+features_folder+'/feat_psnr_for_synth_exp.npy')
#load all bitrate features
all_features_ssim = np.load('./'+features_folder+'/feat_ssim_for_synth_exp.npy')
#load all bitrate features
all_features_vmaf = np.load('./'+features_folder+'/feat_vmaf_for_synth_exp.npy')
#load all bitrate features
all_features_sdn = np.load('./'+features_folder+'/feat_sdn_for_synth_exp.npy')
#load all bitrate features
all_features_logbit = np.load('./'+features_folder+'/feat_logbit_for_synth_exp.npy')
#load all bitrate features
all_features_ftw = np.load('./'+features_folder+'/feat_ftw_for_synth_exp.npy')
#load all videoatlas features
all_features_va = np.load('./'+features_folder+'/feat_va_for_synth_exp.npy')
#load all features iQoE
all_features_iQoE = np.load('./'+features_folder+'/feat_iQoE_for_synth_exp.npy')

def eachuser(user,rs_par):
#for user in range(256):
    maeb = []
    maep = []
    maev = []
    maes = []
    maesdn = []
    maef = []
    maeva = []
    mael = []
    maeiqoe = []
    maeiqoeg = []

    rmseb = []
    rmsep = []
    rmsev = []
    rmseva = []
    rmses = []
    rmsef = []
    rmsesdn = []
    rmsel = []
    rmseiqoe = []
    rmseiqoeg = []

    print('user_'+str(user)+'_rs_'+str(rs_par))
    #splitto i mos
    rs=rs_par
    # bit
    X_train_bit_mos, X_test_bit_mos, mos_train_bit_g, mos_test_bit_g = train_test_split(all_features_bit, mosarray, test_size=0.3,random_state=rs,shuffle=True)
    # psnr
    X_train_psnr_mos, X_test_psnr_mos, mos_train_psnr_g, mos_test_psnr_g = train_test_split(all_features_psnr, mosarray,test_size=0.3, random_state=rs,shuffle=True)
    # vmaf
    X_train_vmaf_mos, X_test_vmaf_mos, mos_train_vmaf_g, mos_test_vmaf_g = train_test_split(all_features_vmaf, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # ssim
    X_train_ssim_mos, X_test_ssim_mos, mos_train_ssim_g, mos_test_ssim_g = train_test_split(all_features_ssim, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # sdn
    X_train_sdn_mos, X_test_sdn_mos, mos_train_sdn_g, mos_test_sdn_g = train_test_split(all_features_sdn, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # ftw
    X_train_ftw_mos, X_test_ftw_mos, mos_train_ftw_g, mos_test_ftw_g = train_test_split(all_features_ftw, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # logbit
    X_train_logbit_mos, X_test_logbit_mos, mos_train_logbit_g, mos_test_logbit_g = train_test_split(all_features_logbit, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # va
    X_train_va_mos, X_test_va_mos, mos_train_va_g, mos_test_va_g = train_test_split(all_features_va, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    # iQoEg
    X_train_iQoE_mos, X_test_iQoE_mos, mos_train_iQoE_g, mos_test_iQoE_g = train_test_split(all_features_iQoE, mosarray,test_size=0.3,random_state=rs,shuffle=True)
    scalermos = MinMaxScaler()
    scalermos.fit(X_train_iQoE_mos)
    X_train_mos = scalermos.transform(X_train_iQoE_mos)
    X_test_mos = scalermos.transform(X_test_iQoE_mos)

    X_train_iQoE_p, X_test_iQoE_p, y_train_iQoE_p, y_test_iQoE_p = train_test_split(all_features_iQoE, users_scores[user],test_size=0.3, random_state=rs,shuffle=True)
    scaler = MinMaxScaler()
    scaler.fit(X_train_iQoE_p)
    X_train_p = scaler.transform(X_train_iQoE_p)
    X_test_p = scaler.transform(X_test_iQoE_p)

    ######train and predict group models
    bitm=fit_linear(X_train_bit_mos,mos_train_bit_g)
    psnrm=fit_linear(X_train_psnr_mos,mos_train_psnr_g)
    vmafm=fit_linear(X_train_vmaf_mos,mos_train_vmaf_g)
    ssimm=fit_linear(X_train_ssim_mos,mos_train_ssim_g)
    sdnm=fit_linear(X_train_sdn_mos,mos_train_sdn_g)
    logbitm = fit_linear(X_train_logbit_mos, mos_train_logbit_g)
    a, b, c, d  = fit_nonlinear((X_train_ftw_mos[:, 0], X_train_ftw_mos[:, 1]),mos_train_ftw_g)
    vam=fit_supreg(X_train_va_mos,mos_train_va_g)
    #predictbitm
    predict_bitm=[]
    for i in X_test_bit_mos:
        predict_bitm.append(np.dot(bitm, i))
    #predictpsnrm
    predict_psnrm=[]
    for i in X_test_psnr_mos:
        predict_psnrm.append(np.dot(psnrm, i))
    #predictvmafm
    predict_vmafm=[]
    for i in X_test_vmaf_mos:
        predict_vmafm.append(np.dot(vmafm, i))
    #predictssimm
    predict_ssimm=[]
    for i in X_test_ssim_mos:
        predict_ssimm.append(np.dot(ssimm, i))
    #predictsdnm
    predict_sdnm=[]
    for i in X_test_sdn_mos:
        predict_sdnm.append(np.dot(sdnm, i))
    #predictlogbitm
    predict_logbitm=[]
    for i in X_test_logbit_mos:
        predict_logbitm.append(np.dot(logbitm, i))

    #predictftwm
    predict_ftwn = []
    for i in X_test_ftw_mos:
        score = a * np.exp(-(b * i[0] + c) * i[1]) + d
        predict_ftwn.append(score)
    #predictvam
    predict_vam=[]
    predict_vam = vam.predict(X_test_va_mos)


    #iqoegroup
    rngmos = np.random.RandomState(42)
    regr_choosen_mos='SVR'
    n_initial_mos = 1
    initial_idx_mos = rngmos.choice(range(len(X_train_mos)), size=n_initial_mos, replace=False)
    X_init_training_mos, y_init_training_mos = X_test_mos[initial_idx_mos], np.array(mos_train_iQoE_g,dtype=int)[initial_idx_mos]

    # Isolate the non-training examples we'll be querying.
    X_pool_mos = np.delete(X_train_mos, initial_idx_mos, axis=0)
    y_pool_mos = np.delete(mos_train_iQoE_g, initial_idx_mos, axis=0)

    Regressors_considered=[RandomForestRegressor(n_estimators = 50, max_depth = 60)]
    Regressors_considered.append(xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1))
    Regressors_considered.append(sklearn.svm.SVR(kernel = 'rbf', gamma= 0.5, C= 100))

    regr_choosen_idx=['RF', 'XGboost', 'SVR','GP'].index(regr_choosen_mos)
    regr_1_mos = Regressors_considered[regr_choosen_idx]

    regressor_gsio_mos = ActiveLearner(
        estimator=regr_1_mos,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training_mos.reshape(-1, 70),
        y_training=y_init_training_mos.reshape(-1, 1).flatten()
    )

    # initial maes
    maes_gsio_mos = [mean_absolute_error(y_test_iQoE_p, regressor_gsio_mos.predict(X_test_mos))]
    rmses_gsio_mos = [np.sqrt(mean_squared_error(y_test_iQoE_p, regressor_gsio_mos.predict(X_test_mos)))]

    X_pool_gsio_mos=X_pool_mos.copy()
    y_pool_gsio_mos=y_pool_mos.copy()

    # active learning
    t_s=10
    count_queries=1
    for idx in range(50):
        #take random queries
        if count_queries<t_s:
            n_samples = len(X_pool_gsio_mos)
            query_idx = rngmos.choice(range(n_samples))

        # gsio
        if count_queries >= t_s:
            query_idx, query_instance = regressor_gsio_mos.query(X_pool_gsio_mos)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio_mos.teach(np.array(X_pool_gsio_mos[query_idx]).reshape(-1, 70),
                           np.array(y_pool_gsio_mos[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio_mos, y_pool_gsio_mos = np.delete(X_pool_gsio_mos, query_idx, axis=0), np.delete(y_pool_gsio_mos, query_idx)

        #save_queries maes
        maes_gsio_mos.append(mean_absolute_error(y_test_iQoE_p, regressor_gsio_mos.predict(X_test_mos)))
        rmses_gsio_mos.append(np.sqrt(mean_squared_error(y_test_iQoE_p, regressor_gsio_mos.predict(X_test_mos))))

        #print('training_query: '+str(count_queries))
        count_queries+=1
    maeiqoeg.append(maes_gsio_mos[-1])
    rmseiqoeg.append(rmses_gsio_mos[-1])

    #maes group
    maeb.append(mean_absolute_error(y_test_iQoE_p, predict_bitm))
    maep.append(mean_absolute_error(y_test_iQoE_p, predict_psnrm))
    maev.append(mean_absolute_error(y_test_iQoE_p, predict_vmafm))
    maes.append(mean_absolute_error(y_test_iQoE_p, predict_ssimm))
    maesdn.append(mean_absolute_error(y_test_iQoE_p, predict_sdnm))
    maef.append(mean_absolute_error(y_test_iQoE_p, predict_ftwn))
    maeva.append(mean_absolute_error(y_test_iQoE_p, predict_vam))
    mael.append(mean_absolute_error(y_test_iQoE_p, predict_logbitm))
    #rmse group
    rmseb.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_bitm)))
    rmsep.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_psnrm)))
    rmsev.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_vmafm)))
    rmses.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_ssimm)))
    rmsef.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_ftwn)))
    rmseva.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_vam)))
    rmsesdn.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_sdnm)))
    rmsel.append(np.sqrt(mean_squared_error(y_test_iQoE_p, predict_logbitm)))


    #iqoe
    ###Active_leanring###
    rng = np.random.RandomState(42)
    regr_choosen='SVR'
    n_initial = 1
    initial_idx = rng.choice(range(len(X_train_p)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train_p[initial_idx], np.array(y_train_iQoE_p,dtype=int)[initial_idx]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train_p, initial_idx, axis=0)
    y_pool = np.delete(y_train_iQoE_p, initial_idx, axis=0)

    Regressors_considered=[RandomForestRegressor(n_estimators = 50, max_depth = 60)]
    Regressors_considered.append(xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1))
    Regressors_considered.append(sklearn.svm.SVR(kernel = 'rbf', gamma= 0.5, C= 100))

    regr_choosen_idx=['RF', 'XGboost', 'SVR','GP'].index(regr_choosen)
    regr_1 = Regressors_considered[regr_choosen_idx]

    regressor_gsio = ActiveLearner(
        estimator=regr_1,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, 70),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )

    # initial maes
    maes_gsio = [mean_absolute_error(y_test_iQoE_p, regressor_gsio.predict(X_test_p))]
    rmses_gsio = [np.sqrt(mean_squared_error(y_test_iQoE_p, regressor_gsio.predict(X_test_p)))]

    X_pool_gsio=X_pool.copy()
    y_pool_gsio=y_pool.copy()

    # active learning
    t_s=10
    count_queries=1
    for idx in range(50):
        #take random queries
        if count_queries<t_s:
            n_samples = len(X_pool_gsio)
            query_idx = rng.choice(range(n_samples))

        # gsio
        if count_queries >= t_s:
            query_idx, query_instance = regressor_gsio.query(X_pool_gsio)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, 70),
                           np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)

        #save_queries maes
        maes_gsio.append(mean_absolute_error(y_test_iQoE_p, regressor_gsio.predict(X_test_p)))
        rmses_gsio.append(np.sqrt(mean_squared_error(y_test_iQoE_p, regressor_gsio.predict(X_test_p))))

        #print('training_query: '+str(count_queries))
        count_queries+=1
    maeiqoe.append(maes_gsio[-1])
    rmseiqoe.append(rmses_gsio[-1])

    #save all
    fold='save_all_models_users/'
    np.save(fold+'maeb_'+str(user)+'_'+str(rs)+'.npy', maeb)
    np.save(fold+'maep_'+str(user)+'_'+str(rs)+'.npy', maep)
    np.save(fold+'maev_'+str(user)+'_'+str(rs)+'.npy', maev)
    np.save(fold+'maes_'+str(user)+'_'+str(rs)+'.npy', maes)
    np.save(fold+'maef_'+str(user)+'_'+str(rs)+'.npy', maef)
    np.save(fold+'maeva_'+str(user)+'_'+str(rs)+'.npy', maeva)
    np.save(fold+'maesdn_'+str(user)+'_'+str(rs)+'.npy', maesdn)
    np.save(fold+'mael_'+str(user)+'_'+str(rs)+'.npy', mael)
    np.save(fold+'maeiqoe_'+str(user)+'_'+str(rs)+'.npy', maeiqoe)
    np.save(fold+'maeiqoeg_'+str(user)+'_'+str(rs)+'.npy', maeiqoeg)

    np.save(fold+'rmseb_'+str(user)+'_'+str(rs)+'.npy', rmseb)
    np.save(fold+'rmsep_'+str(user)+'_'+str(rs)+'.npy', rmsep)
    np.save(fold+'rmsev_'+str(user)+'_'+str(rs)+'.npy', rmsev)
    np.save(fold+'rmses_'+str(user)+'_'+str(rs)+'.npy', rmses)
    np.save(fold+'rmsef_'+str(user)+'_'+str(rs)+'.npy', rmsef)
    np.save(fold+'rmseva_'+str(user)+'_'+str(rs)+'.npy', rmseva)
    np.save(fold+'rmsesdn_'+str(user)+'_'+str(rs)+'.npy', rmsesdn)
    np.save(fold+'rmsel_'+str(user)+'_'+str(rs)+'.npy', rmsel)
    np.save(fold+'rmseiqoe_'+str(user)+'_'+str(rs)+'.npy', rmseiqoe)
    np.save(fold+'rmseiqoeg_'+str(user)+'_'+str(rs)+'.npy', rmseiqoeg)



if __name__ == "__main__":
    from multiprocessing import Pool
    fold = 'save_all_models_users/'
    comb_of_par = []
    for user in range(256):
        for rs in [42, 13, 70, 34, 104]:
            if not os.path.exists(fold+'rmseiqoeg_'+str(user)+'_'+str(rs)+'.npy'):
                comb_of_par.append((user, rs))
    print('params nr_'+str(len(comb_of_par)))
    with Pool() as p:
        #p.map(each_user, [u for u in range(32)])
        p.starmap(eachuser, comb_of_par)
    p.close()

    fold = 'save_all_models_users/'
    save_fold='save_all_models_ave_users/'
    #mae
    for model in ['maeb','maep','maev','maes','maesdn','maef','maeva','mael','maeiqoe','maeiqoeg']:
        m=[]
        st=[]
        for user in range(256):
            r=[]
            for rs in [42, 13, 70, 34, 104]:
                r.append(np.load(fold+model+'_'+str(user)+'_'+str(rs)+'.npy'))
            np.save(save_fold+model+'_'+str(user)+'_maeave.npy',np.mean(r))
            np.save(save_fold+model + '_' + str(user) + '_maestd.npy', np.std(r))

    #rmse
    save_aves_rmse=[]
    for model in ['rmseb','rmsep','rmsev','rmses','rmsef','rmseva','rmsesdn','rmsel','rmseiqoe','rmseiqoeg']:
        m = []
        st = []
        for user in range(256):
            r = []
            for rs in [42, 13, 70, 34, 104]:
                r.append(np.load(fold + model + '_' + str(user) + '_' + str(rs) + '.npy'))
            np.save(save_fold+model + '_' + str(user) + '_rmseave.npy', np.mean(r))
            np.save(save_fold+model + '_' + str(user) + '_rmsestd.npy', np.std(r))



