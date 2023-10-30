## Generate Figure 16:

### LA
* Input data: folders save_all_models_ave_users and save_all_models_users which contain data for the 8 baselines models considered, sota_results which contain the data for p1203 and LSTM models
* Run generate_figure_16.py to generate figure 16
* Run Plot_legend to generate the legend for figure 16

### LB
* Input data: folder features_generated_experiences, synthetic_users_scores_for_generated_experiences, p1203_scores.npy, scoresbiqps.npy. The first two can be generated in Synthetic_raters, the last two can be found in Synthetic_raters/p1203_and_LSTM_models 
* Run generate_figure_16.py to generate figure 16 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folders save_all_models_ave_users and save_all_models_users which contain data for the 8 baselines models considered
  * folder sota_results which contain the data for p1203 and LSTM models
* Run Plot_legend to generate the legend for figure 16
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some hours are needed. 

