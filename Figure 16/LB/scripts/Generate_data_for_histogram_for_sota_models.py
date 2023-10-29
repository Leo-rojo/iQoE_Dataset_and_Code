import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os

features_folder='../input_data/features_generated_experiences'
p1203scores=np.load('../input_data/p1203_scores.npy')
bilstmscores=np.load('../input_data/scoresbiqps.npy')
all_features_va = np.load('./'+features_folder+'/feat_va_for_synth_exp.npy')
users_scores=np.load('../input_data/synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')
users_scores=users_scores.reshape(256,1000)
#run for 5 different seeds for reducing the impact of random shuffles
for rs in [42,13,70,34,104]:
    #split train test for group models, iQoE-group and iQoE-personal
    X_train_p1203, X_test_p1203, y_train_p1203, y_test_p1203 = train_test_split(all_features_va, p1203scores, test_size=0.3, random_state=rs,shuffle=True) #all_features_va is just a placeholder for the train_test_split function
    X_train_bilstm, X_test_bilstm, y_train_bilstm, y_test_bilstm = train_test_split(all_features_va, bilstmscores, test_size=0.3, random_state=rs,shuffle=True) #all_features_va is just a placeholder for the train_test_split function

    tr_te_each_users=[]
    for u in range(256):
        X_train_iQoE_p, X_test_iQoE_p, y_train_iQoE_p, y_test_iQoE_p = train_test_split(all_features_va, users_scores[u], test_size=0.3, random_state=rs,shuffle=True) #all_features_va is just a placeholder for the train_test_split function
        tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])

    #p1203 prediction on test
    mosmodel_us_scores_mae_p1203=[]
    mosmodel_us_scores_rmse_p1203=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_p1203=y_test_p1203
        mosmodel_us_scores_mae_p1203.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_p1203))
        mosmodel_us_scores_rmse_p1203.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_p1203)))

    if not os.path.exists('../output_data/sota_results/p1203_scores'):
        os.makedirs('../output_data/sota_results/p1203_scores')
    np.save('../output_data/sota_results/p1203_scores/mae'+str(rs),mosmodel_us_scores_mae_p1203)
    np.save('../output_data/sota_results/p1203_scores/rmse' + str(rs), mosmodel_us_scores_rmse_p1203)

    #bilstm prediction on test
    mosmodel_us_scores_mae_bilstm=[]
    mosmodel_us_scores_rmse_bilstm=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_bilstm=y_test_bilstm
        mosmodel_us_scores_mae_bilstm.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_bilstm))
        mosmodel_us_scores_rmse_bilstm.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_bilstm)))
    if not os.path.exists('../output_data/sota_results/bilstm_scores'):
        os.makedirs('../output_data/sota_results/bilstm_scores')
    np.save('../output_data/sota_results/bilstm_scores/mae'+str(rs),mosmodel_us_scores_mae_bilstm)
    np.save('../output_data/sota_results/bilstm_scores/rmse' + str(rs), mosmodel_us_scores_rmse_bilstm)


#aggregate results for the 5 different shuffles
sc=['p1203','bilstm']
for metric in ['mae','rmse']:
    for score in sc:
        each=[]
        for rs in [42,13,70,34,104]:
            each.append(np.load('../output_data/sota_results/'+score+'_scores/'+metric+str(rs)+'.npy'))
        m=np.mean(each,axis=0)
        std=np.std(each,axis=0)
        np.save('../output_data/sota_results/'+score+'_scores/'+metric+'_ave',m)
        np.save('../output_data/sota_results/' + score + '_scores/'+metric+'_std', std)