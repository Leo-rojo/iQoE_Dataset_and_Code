import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Generate_data_for_histogram_for_sota_models.py')
print('generate data for p1203 and LSTM models done')
os.system('python Generate_data_for_histogram.py')
print('generate data for baselines models done')
os.system('python plot_figure_16.py')
print('generate figure 16 done')











