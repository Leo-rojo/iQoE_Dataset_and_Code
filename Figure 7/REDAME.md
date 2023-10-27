## Generate Figure 7
* input files: users folder with 128 users (8 more than for the other experiments) + p1203_mae_each_user.npy, p1203_rmse_each_user.npy, lstm_mae_each_user.npy,lstm_rmse_each_user.npy, iqoe_mae_each_user.npy, iqoe_rmse_each_user.npy. All these files have been generated for producing Figure 6 and 8 and can be found in folder results_collection of Figure 6 and 8 and Table 1.
* Run Generate_features_and_scores.py to generate the features_qoes_test and features_qoes_train folders plus test_scores.npy and baselines_scores.npy files.
* Run Generate_different_group_inclusion.py to generate Figure 7 and more detailed results in .xls format
* Run Plot_legend_different_group_inclusion.py to generate the legend of Figure 7