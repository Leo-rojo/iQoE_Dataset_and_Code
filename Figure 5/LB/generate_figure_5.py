import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python generate_dataset.py')
print('Dataset done')

print('Figure a done')
os.system('python scores_distribution.py')
print('Figure a done')
os.system('python time_inconsistencies.py')
print('Figure b done')
os.system('python calculate_features_for_iqoe.py')
print('iqoe features done')
os.system('python read_the_stall_for_each_exp_of_each_user.py')
print('calculate times for figure c done')
#run Elaborate_results_group_qoes.py
os.system('python Times.py')
print('Figure c done')










