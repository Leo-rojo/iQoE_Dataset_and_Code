import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Generate_allML_rigs.py')
print('modeler designs with rigs done')
os.system('python Plot_continuous_metrics_across_ML.py')
print('query evolution plot done ')
os.system('python Plot_ECDF_50SA.py')
print('ecdf plots done')









