import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python Generate_points_user_4.py')
print('Generate data points done')
os.system('python Plot_points.py')
print('Plot done')









