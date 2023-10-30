import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python Figure_a.py')
print('Figure a done')
os.system('python Figure_b.py')
print('Figure b done')
#run Elaborate_results_group_qoes.py
os.system('python Figure_c.py')
print('Figure c done')










