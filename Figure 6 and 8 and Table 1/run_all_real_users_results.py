import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#insert path to your folder Figure 6 and 8 and Table 1

os.system('python generate_dataset.py')
print('step0 done')
os.system('python calculate_features_for_qoes.py')
print('step1 done')
#run Elaborate_results_group_qoes.py
os.system('python Elaborate_results_group_qoes.py')
print('step2 done')
#run Elaborate_results_iQoE_test.py
os.system('python Elaborate_results_iQoE_test.py')
print('step3 done')
#run Elaborate_results_personalized_qoes.py
os.system('python Elaborate_results_personalized_qoes.py')
print('step4 done')
#run mae_rmse_sota_models.py
os.system('python sotamodels/Test_videos_iqoe/mae_rmse_sota_models.py')
print('step5 done')

#plot ecdf
os.system('python generate_figure_6.py')
os.system('python plot_legend_dashed.py')
print('plot ecdf done')
##ecdf details
#run print_gaining_factors_and_iqoe_for_each_users.py
os.system('python print_gaining_factors_and_iqoe_for_all_users_median.py')
#run print_gaining_factors_and_iqoe_for_each_users.py
os.system('python print_gaining_factors_and_iqoe_for_each_users_median_personal.py')
print('detailed results done')

#load.xlsx files in dataframe
import pandas as pd
# Load the Excel file into a DataFrame
df = pd.read_excel('gain_factors_mae_all_users_median.xlsx')
# Extract the last two rows into a new DataFrame
last_two_rows = df.tail(2)
#remove last two columns
last_two_rows = last_two_rows.iloc[:, :-2]
#remove first column
last_two_rows_mae = last_two_rows.iloc[:, 1:]

#do the same for rmse
df = pd.read_excel('gain_factors_rmse_all_users_median.xlsx')
# Extract the last two rows into a new DataFrame
last_two_rows = df.tail(2)
#remove last two columns
last_two_rows = last_two_rows.iloc[:, :-2]
#remove first column
last_two_rows_rmse = last_two_rows.iloc[:, 1:]

print('all')
titlerow=last_two_rows_mae.columns.tolist()
maerow=last_two_rows_mae.iloc[0].tolist()
rmserow=last_two_rows_rmse.iloc[0].tolist()
print([x for _,x in sorted(zip(maerow,titlerow))])
print(sorted(maerow))
print([x for _,x in sorted(zip(maerow,rmserow))])
print('10%')
titlerow_ten=last_two_rows_mae.columns.tolist()
maerow_ten=last_two_rows_mae.iloc[1].tolist()
rmserow_ten=last_two_rows_rmse.iloc[1].tolist()
print([x for _,x in sorted(zip(maerow,titlerow_ten))])
print([x for _,x in sorted(zip(maerow,maerow_ten))])
print([x for _,x in sorted(zip(maerow,rmserow_ten))])

import math
#count nur of folders in users_cleaned
path_iQoE='users'
identifiers = []
conta=0
for fold in os.listdir(path_iQoE):
    conta=conta+1
ten_percent=math.ceil((conta-2)*10/100)
print(ten_percent)

#values ecdf
leg = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas','p1203', 'LSTM','iQoE'] #this is the order of ecdfs
path_ecdf='Plot_ECDF'
import numpy as np
maeecdf=np.load(path_ecdf+'/values_ecdfmae.npy',allow_pickle=True)
rmsecdf=np.load(path_ecdf+'/values_ecdfrmse.npy',allow_pickle=True)

value_tresh_ecdf=0.2
#lowest values for mae and rmse iqoe
mae_iqoe=maeecdf[-1]
rmse_iqoe=rmsecdf[-1]
print('min mae iqoe',mae_iqoe.x[1])
print('min rmse iqoe',rmse_iqoe.x[1])
#thake the value of mae_iqoe.x that correspond to the element of mae_iqoe.y that is closer to 0.8
mae_iqoe_02=mae_iqoe.x[np.argmin(np.abs(mae_iqoe.y - value_tresh_ecdf))]
rmse_iqoe_02=rmse_iqoe.x[np.argmin(np.abs(rmse_iqoe.y - value_tresh_ecdf))]
print('mae iqoe 0.2',mae_iqoe_02)
print('rmse iqoe 0.2',rmse_iqoe_02)
#same for videoAtlas
mae_videoAtlas=maeecdf[-4]
rmse_videoAtlas=rmsecdf[-4]
print('min mae videoAtlas',mae_videoAtlas.x[1])
print('min rmse videoAtlas',rmse_videoAtlas.x[1])
#thake the value of mae_iqoe.x that correspond to the element of mae_iqoe.y that is equal to 0.8
mae_videoAtlas_02=mae_videoAtlas.x[np.argmin(np.abs(mae_videoAtlas.y - value_tresh_ecdf))]
rmse_videoAtlas_02=rmse_videoAtlas.x[np.argmin(np.abs(rmse_videoAtlas.y - value_tresh_ecdf))]
print('mae videoAtlas 0.2',mae_videoAtlas_02)
print('rmse videoAtlas 0.2',rmse_videoAtlas_02)
#print gain factors iqoe respect va
print('gain factors iqoe respect va')
print('mae',mae_videoAtlas_02/mae_iqoe_02)
print('rmse',rmse_videoAtlas_02/rmse_iqoe_02)









