## Generate Figure 13:

### LA
* Input data: folders SVR_results_qn_60_nr_ch_7_1 which is the folder containing the iQoE performance when trained and tested on different ABRs
* Run generate_figure_13.py to generate figure 13
* Run Plot_legend to generate the legend for figure 13

### LB
* Input data: Fitted_models_without_logistic, save_param_sigmoids, experiences_with_features.npy. The files can be generated from Synthetic_users_implementation
* Run generate_figure_13.py to generate figure 13 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folder exp_and_scores_each_ABR which contains the videostreaming experiences and the synthetic scores associated for each ABRs
  * folder SVR_results_qn_60_nr_ch_7_1 which is the folder containing the iQoE performance when trained and tested on different ABRs
  * Run Plot_legend to generate the legend for figure 13
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some hours are needed. 
