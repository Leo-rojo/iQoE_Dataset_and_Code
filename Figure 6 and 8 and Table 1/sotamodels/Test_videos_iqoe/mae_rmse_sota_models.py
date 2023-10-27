import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#os.chdir('/Figure 6 and 8 and Table 1')
path_iQoE='users'

sota_folder='sotamodels\\Test_videos_iqoe'
p1203_scores=np.load(sota_folder+'/p1203_scores.npy')
lstm_scores=np.load(sota_folder+'/scoresbiqps.npy')

maes_p1203=[]
rmses_p1203=[]
maes_lstm=[]
rmses_lstm=[]
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

        idx_sota_scores = list(np.load(path_iQoE + '/original_database/idx_col_test.npy'))
        idx_y_test=list(d_test.keys())
        p1203_aligned=[]
        lstm_aligned=[]
        for i in idx_y_test:
            idx_in_sota=idx_sota_scores.index(i)
            p1203_aligned.append(p1203_scores[idx_in_sota])
            lstm_aligned.append(lstm_scores[idx_in_sota])



        maes_p1203.append(mean_absolute_error(y_test, p1203_aligned))
        rmses_p1203.append(sqrt(mean_squared_error(y_test, p1203_aligned)))
        maes_lstm.append(mean_absolute_error(y_test, lstm_aligned))
        rmses_lstm.append(sqrt(mean_squared_error(y_test, lstm_aligned)))

save_path='results_collection\\'
np.save(save_path+'p1203_mae_each_user.npy',maes_p1203)
np.save(save_path+'p1203_rmse_each_user.npy',rmses_p1203)
np.save(save_path+'lstm_mae_each_user.npy',maes_lstm)
np.save(save_path+'lstm_rmse_each_user.npy',rmses_lstm)
print('-----------test_done-----------')