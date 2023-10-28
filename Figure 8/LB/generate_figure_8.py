import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
#insert path to your folder Figure 6 and 8 and Table 1
os.system('python calculate_features_for_qoes.py')
print('features for models done')
#run Elaborate_results_group_qoes.py
os.system('python Elaborate_results_group_qoes.py')
print('step2 done')
#run Elaborate_results_iQoE_test.py
os.system('python Elaborate_results_iQoE_test.py')
print('step3 done')

os.system('python Elaborate_results_personalized_qoes.py')
print('step4 done')

#run mae_rmse_sota_models.py
os.system('python sotamodels/Test_videos_iqoe/mae_rmse_sota_models.py')
print('step5 done')

#plot ecdf
os.system('python plot_fig.py')
os.system('python plot_legend_dashed.py')
print('plot ecdf done')
##ecdf details