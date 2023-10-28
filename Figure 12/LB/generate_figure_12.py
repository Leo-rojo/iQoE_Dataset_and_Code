import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Generate_results_iQoE_across_different_h.py')
print('generate results for different values of h')
os.system('python Plot_continuous_metrics_across_different_h.py')
print('Figure 12 done')











