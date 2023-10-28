import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Plot_queries_evolution_for_samplingstrategies_plus_XSVR.py')
print('query evolution plot done ')
os.system('python Plot_ECDF_50SA_for_samplingstrategies_plus_XSVR.py')
print('ecdf plots done')









