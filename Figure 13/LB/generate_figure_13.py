import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python experiences_abrs.py')
print('generate experiences from different abrs done')
os.system('python Generate_results_for_abr_generalizability.py')
print('generate results for figure 13 done')
os.system('python Plot_differentabr.py')
print('generate figure 13 done')











