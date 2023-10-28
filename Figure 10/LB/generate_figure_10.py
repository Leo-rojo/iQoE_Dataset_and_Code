import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Generate_combination_of_sampling_strategies_plus_modelers_with_initial_random.py')
print('sampler designs with initial random done')
os.system('python Generate_combination_of_sampling_strategies_plus_modelers_without_initial_random.py')
print('sampling designs without initial random done')
os.system('python Plot_queries_evolution_for_samplingstrategies_plus_XSVR.py')
print('query evolution plot done ')
os.system('python Plot_ECDF_50SA_for_samplingstrategies_plus_XSVR.py')
print('ecdf plots done')









