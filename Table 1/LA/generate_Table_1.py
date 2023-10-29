import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

tot_users=120

os.chdir('scripts')
#run print_gaining_factors_and_iqoe_for_each_users.py
os.system('python print_gaining_factors_and_iqoe_for_all_users_median.py')
#run print_gaining_factors_and_iqoe_for_each_users.py
os.system('python print_gaining_factors_and_iqoe_for_each_users_median_personal.py')
print('detailed results done')
os.chdir('..')

#load.xlsx files in dataframe
import pandas as pd
# Load the Excel file into a DataFrame
df = pd.read_excel('output_data/gain_factors_mae_all_users_median.xlsx')
# Extract the last two rows into a new DataFrame
last_two_rows = df.tail(2)
#remove last two columns
last_two_rows = last_two_rows.iloc[:, :-2]
#remove first column
last_two_rows_mae = last_two_rows.iloc[:, 1:]

#do the same for rmse
df = pd.read_excel('output_data/gain_factors_rmse_all_users_median.xlsx')
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









