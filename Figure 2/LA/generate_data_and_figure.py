import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python Save_features_and_rater_scores_W4.py')
print('features and scores from W4 done')
os.system('python Generate_videoatlas_features.py')
print('features for videoatlas model done')
os.system('python Figure_a.py')
print('Figure a done')
os.system('python Figure_b.py')
print('Figure b done')
#run Elaborate_results_group_qoes.py
os.system('python Figure_c.py')
print('Figure c done')










