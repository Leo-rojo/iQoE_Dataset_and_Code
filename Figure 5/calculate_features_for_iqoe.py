import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os

#insert path to your folder Figure 5

nr_c = 4
path_iQoE='users'
maes=[]
rmses=[]
scores_more_users_pers_base=[]
videos_more_users_pers_base=[]
scores_more_users_test=[]
videos_more_users_test=[]
exp_orig_train_all_users=[]
exp_orig_test_all_users=[]
exp_orig_all_all_users=[]
y_all_all_users=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier

        ##personal data
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
        a=[int(i) for i in list(d_train.values())]
        a2=[int(i) for i in list(d_train.keys())]

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
        b=[int(i) for i in list(d_baselines.values())]
        b2=[int(i) for i in list(d_baselines.keys())]

        #test data
        c=[]
        c2=[]
        d_test = {}
        with open(user_folder + '/Scores_test_' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_test[int(key)] = val
        c=[int(i) for i in list(d_test.values())]
        c2=[int(i) for i in list(d_test.keys())]

        scores_more_users_pers_base.append(a + b)
        videos_more_users_pers_base.append(a2 + b2)
        scores_more_users_test.append(c)
        videos_more_users_test.append(c2)

        idx_col_train = np.load(path_iQoE + '/original_database/idx_col_train.npy')
        exp_orig = np.load(path_iQoE + '/original_database/synth_exp_train.npy')
        scaled_exp_orig = np.load(path_iQoE + '/original_database/X_train_scaled.npy')
        exp_orig_train = []
        scaled_exp_orig_train = []
        orig_orig = []
        merge_dictionary = a2+b2#{**d_train, **d_baselines}
        print(identifier)
        print(len(a2+b2))
        for x in merge_dictionary:  # array of original idxs
            idx_standard = np.where(idx_col_train == x)
            exp_orig_train.append(exp_orig[idx_standard])
            scaled_exp_orig_train.append(scaled_exp_orig[idx_standard])
        #print(exp_orig_train)


        exp_orig_test = []
        scaled_exp_orig_test = []
        idx_col_test = np.load(path_iQoE + '/original_database/idx_col_test.npy')
        exp_orig_te = np.load(path_iQoE + '/original_database/synth_exp_test.npy')
        scaled_exp_orig = np.load(path_iQoE + '/original_database/X_test_scaled.npy')
        for x in d_test.keys():  # array of original idxs
            idx_standard = np.where(idx_col_test == x)
            exp_orig_test.append(exp_orig_te[idx_standard])
            scaled_exp_orig_test.append(scaled_exp_orig[idx_standard])

        exp_orig_all_all_users.append(exp_orig_train+exp_orig_test)
        y_all_all_users.append(a+b+c)

#convert exp_orig_all_all_users in list of lists
exp_orig_all_all_users_list=[]
for user in exp_orig_all_all_users:
    exp_orig_all_all_users_list.append([x[0][:40].tolist() for x in user])

np.save('feat_iqoe.npy', exp_orig_all_all_users_list)