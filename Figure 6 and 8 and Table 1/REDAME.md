## Generate Figure 6 and 8
* input files: folder users containing all the users information, sotamodels folder containing maes and rmses of p1203 and LSTM models for the test set.
* Run run_all_real_users_results.py to generate the ECDF plots in Figure 6 and 8 plus .xls files containing detailed results. The file runs in order:
  * calculate_features_for_qoes.py: it calculates the training and test dataset features for the 8 QoE models B, G, V, R, S, N, F, and A.
  * Elaborate_results_group_qoes.py: it generates maes and rmses metrics for the test set for all the 8 models trained with MOS scores
  * Elaborate_results_iQoE_test.py: it generates maes and rmses metrics for the test set for the iQoE model
  * Elaborate_results_personalized_qoes.py: it generates maes and rmses metrics for the test set for 8 models retrained with individual scores
  * sotamodels/mae_rmse_sota_models.py: it generates maes and rmses metrics for the test set for the p1203 and LSTM models
  * plot_ECDF_dashed.py: it generates the ECDFs plots in Figure 5 and 7
  * Plot_legend_dashed.py: it generates the legends for the ECDFs plots in Figure 5 and 7
  * print_gaining_factors_and_iqoe_for_all_users_median.py: it generates detailed results for group_models trained with MOS
  * print_gaining_factors_and_iqoe_for_each_users_median_personal.py: it generates detailed results for group_models trained with individual scores
  * it prints on screen the values for Table 1