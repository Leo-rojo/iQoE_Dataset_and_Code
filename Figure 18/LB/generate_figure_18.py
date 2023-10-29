import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python Generate_results_for_time_and_space_overhead.py')
print('generate data for calculating time and space overhead done')
os.system('python Plots.py')
print('figure 18 done')









