import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

maes=[]
rmses=[]
maes_rnd=[]
rmses_rnd=[]
path_iQoE= '../input_data/users'#insert path to your folder Figure 15/users
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

        chunks_names=['c_1','c_2','c_3','c_4']
        column_n= ['rep_index','rebuffering_duration','video_bitrate','chunk_size','width','height','is_best','psnr','ssim','vmaf']
        column_names=[]
        for i in chunks_names:
            for k in column_n:
                column_names.append(i+'_'+k)
        column_names.append('score')

        # Calling DataFrame constructor on list
        df = pd.DataFrame(rft,columns=column_names)

        #create folder dataset if it does not exist
        if not os.path.exists('../output_data/dataset'):
            os.makedirs('../output_data/dataset')
        #save df to excel file
        df.to_excel('../output_data/dataset/user_'+identifier+'.xlsx', index=False)


