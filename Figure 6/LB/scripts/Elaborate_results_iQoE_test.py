import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
#insert path to your folder Figure 6 and 8 and Table 1
maes=[]
rmses=[]
path_iQoE='../input_data/users'
y_predicted_by_user=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        #print(identifier)
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


        #iQoE mae and rmse
        all_m = [int(lastmodel.split('.')[0]) for lastmodel in os.listdir(user_folder + '/models_' + identifier)]
        sorted_models = sorted(all_m)
        m=sorted_models[-1]
        lm = user_folder + '/models_' +identifier + '/' + str(m) + '.pkl'
        with open(lm, 'rb') as file:
            reg = pickle.load(file)
        a=len(scaled_exp_orig_test)
        b=len(scaled_exp_orig_test[0][0])
        X_test=np.array(scaled_exp_orig_test).reshape(a,b)

        y_predicted_by_user.append(reg.predict(X_test))


        # scores iQoE
        maes.append(mean_absolute_error(y_test, reg.predict(X_test)))
        rmses.append(sqrt(mean_squared_error(y_test, reg.predict(X_test))))

#print(maes)
#print(rmses)
np.save('../output_data/iQoE_mae_each_user',maes)
np.save('../output_data/iQoE_rmse_each_user',rmses)




