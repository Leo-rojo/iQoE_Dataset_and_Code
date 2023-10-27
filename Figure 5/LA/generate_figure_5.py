import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python scores_distribution.py')
print('Figure a done')
os.system('python time_inconsistencies.py')
print('Figure b done')
#run Elaborate_results_group_qoes.py
os.system('python times.py')
print('Figure c done')










