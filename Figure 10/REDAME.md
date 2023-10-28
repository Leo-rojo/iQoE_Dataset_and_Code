## Generate Figure 10:

### LA
* Input data: SVR_results_qn_250_nr_ch_7_1no_R that contains data for different sampling strategies with initial random search, SVR_results_qn_250_nr_ch_7_1R that data for different sampling strategies without initial random search
* Run generate_figure_10.py to generate figure 10
* Run Plot_legend to generate the legend for figure 10

### LB
* Input data: folder features_generated_experiences and synthetic_users_scores_for_generated_experiences. Both of them can be generated in Synthetic_users_implementation.
* Run generate_figure_10.py to generate figure 10 and run all the other scripts contained in scripts. Those files produce in output folder:
  * SVR_results_qn_250_nr_ch_7_1no_R: data for different sampling strategies with initial random search
  * SVR_results_qn_250_nr_ch_7_1R: data for different sampling strategies without initial random search
  * Run Plot_legend to generate the legend for figure 10
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some hours are needed. 