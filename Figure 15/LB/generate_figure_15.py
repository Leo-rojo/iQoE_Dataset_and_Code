import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python generate_dataset_formatted.py')
print('generate dataset formatted done')
os.system('python IFs_resolution.py')
print('generate figure 15a done')
os.system('python scores_distribution_fig15_ecdf.py')
print('generate figure 15b done')











