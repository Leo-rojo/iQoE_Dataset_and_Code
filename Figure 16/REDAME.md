## Generate Figure 16:
* input file: folder features_generated_experiences, synthetic_users_scores_for_generated_experiences, p1203_scores.npy, scoresbiqps.npy. The first two can be generated in Synthetic_users_implementation, the last two can be found in Synthetic_users_implementation/p1203_and_LSTM_models 
* Run Generate_data_for_histogram.py, it will save data for plots in save_all_models_ave_users and save_all_models_users for the 8 baselines models considered
* Run Generate_data_for_histogram_for_sota_models.py, it will save the data for plots in sota_results for p1203 and LSTM models
* Run Plot_histogram.py to generate Figure 16
* Run Plot_legend.py to generate the legend for Figure 16