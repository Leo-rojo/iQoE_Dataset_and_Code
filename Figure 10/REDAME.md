## Generate Figure 10:
* input file: folder features_generated_experiences and synthetic_users_scores_for_generated_experiences. Both of them can be generated in Synthetic_users_implementation.
* Run select_nr_of_cluster.py for selecting the right number of cluster for cluster uncertainty sampling strategy using the elbow method
* Run Generate_combination_of_sampling_strategies_plus_modelers_with_initial_random.py in order to generate data for different sampling strategies with initial random search and different modelers and save in new generated folders with name 'modeler_results_qn_nr_ch_7_1R'
* Run Generate_combination_of_sampling_strategies_plus_modelers_without_initial_random.py in order to generate data for different sampling strategies without initial random search and different modelers and save in new generated folders with name 'modeler_results_qn_nr_ch_7_1no_R'
* Run Plot_queries_evolution_for_samplingstrategies_plus_XSVR.py and the plots will be saved in new folder Plot_continuous_metrics_SVRynR
* Run Plot_ECDF_50SA_for_samplingstrategies_plus_XSVR.py and plots will be saved in new folder Plot_ECDF
