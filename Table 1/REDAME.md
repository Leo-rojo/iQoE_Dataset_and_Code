## Generate Table 1

### LA_&_LB
* Input data: folder 'users' which contains info of the 120 users that took the assesment, dataset folder containing the well formatted .xls dataset, mae and rmse results for each users for the 8 baselines, iQoE, p1203, LSTM and personalized baselines as .npy files
* Run generate_Table_1.py to print the values of TAble 1 and run all the other scripts contained in scripts. Those files produce in output folder:
  * the gain factors of iQoE respect to the 8 baselines trained with MOS plus LSTM and P1203 models as .xlsx files
  * the gain factors of iQoE respect to the 8 baselines trained with individual users scores plus LSTM and P1203 models as .xlsx files

