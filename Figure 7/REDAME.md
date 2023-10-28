## Generate Figure 7

### LA
* Input data: features_qoes_test and features_qoes_train, baselines_scores.npy, identifiers_order.npy, test_scores.npy,p1203_mae_each_user.npy, p1203_rmse_each_user.npy, lstm_mae_each_user.npy,lstm_rmse_each_user.npy, iqoe_mae_each_user.npy, iqoe_rmse_each_user.npy
* Run generate_figure_7.py to generate figure 7
* Run Plot_legend_different_group_inclusion.py to generate the legend of Figure 7

### LB
* Input data: users folder with 128 users (8 more than for the other experiments) + p1203_mae_each_user.npy, p1203_rmse_each_user.npy, lstm_mae_each_user.npy,lstm_rmse_each_user.npy, iqoe_mae_each_user.npy, iqoe_rmse_each_user.npy. All these files have been generated for producing Figure 6 
* Run generate_figure_7.py to generate figure 7 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folders features_qoes_test and features_qoes_train 
  * baselines_scores.npy, identifiers_order.npy, test_scores.npy
* Run Plot_legend_different_group_inclusion.py to generate the legend of Figure 7