import numpy as np
import os

#os.chdir('insert path to Figure 11\\Plot_continuous_metrics_RiGS')
leg = ['iQoE','iGS+XGB','iGS+RF','iGS+GP']
maes=np.load('maevalues.npy',allow_pickle=True)