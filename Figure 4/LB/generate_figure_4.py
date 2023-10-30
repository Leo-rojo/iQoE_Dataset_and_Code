import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')

os.system('python features_hdtv_blocks_permutation.py')
print('features values done')
os.system('python generate_figure.py')
print('Plot done')