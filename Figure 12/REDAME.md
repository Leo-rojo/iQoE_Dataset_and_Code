## Generate Figure 12:

### LA
* Input data: folders SVRigs_results_qn_250_nr_ch_7_h which are folders containing the results for different values of h parameter for iQoE method
* Run generate_figure_12.py to generate figure 12
* Run Plot_legend to generate the legend for figure 12

### LB
* Input data: folder features_generated_experiences and synthetic_users_scores_for_generated_experiences. Both of them can be generated in Synthetic_users_implementation.
* Run generate_figure_12.py to generate figure 12 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folders SVRigs_results_qn_250_nr_ch_7_h which are folders containing the results for different values of h parameter for iQoE method
  * Run Plot_legend to generate the legend for figure 12
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some hours are needed. 
