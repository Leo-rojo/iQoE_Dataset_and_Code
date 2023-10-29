import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Genereate_results_for_different_train_test_split_01_09.py')
print('generate data for train test split from 0 to 0.1 done')
os.system('python Genereate_results_for_different_train_test_split_0_01.py')
print('generate data for train test split from 0.1 to 0.9 done')
os.system('python plot_figure.py')
print('generate figure 17 done')











