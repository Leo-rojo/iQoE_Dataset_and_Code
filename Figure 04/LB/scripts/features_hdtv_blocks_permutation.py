import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
from sklearn import svm

#insert path to your folder

#import features and scores for hdtv
all_features_hdtv=np.load('../input_data/features_hdtv.npy')
users_score_hdtv=np.load('../input_data/users_raters_hdtv.npy')

def move_columns(arr, current_indices, new_indices):
    # Create a copy of the array
    new_arr = arr.copy()

    # Ensure the current_indices and new_indices have the same length
    if len(current_indices) != len(new_indices):
        raise ValueError("The current_indices and new_indices must have the same length.")

    # Move the columns to the new positions
    for curr, new in zip(current_indices, new_indices):
        if curr < arr.shape[1] and new < arr.shape[1]:
            new_arr[:, [new, curr]] = new_arr[:, [curr, new]]

    return new_arr

#remove from all_features_hdtv coloumn 3,6,9,16,18,19...
column_to_remove_dur=[i for i in range(3,len(all_features_hdtv[0]),13)]
column_to_remove_qp=[i for i in range(5,len(all_features_hdtv[0]),13)]
column_to_remove_framerate=[i for i in range(6,len(all_features_hdtv[0]),13)]
all_features_hdtv=np.delete(all_features_hdtv, column_to_remove_framerate+column_to_remove_qp+column_to_remove_dur, axis=1)
#group features
all_features_hdtv_grouped=[]
for k in range(len(all_features_hdtv)):
    all_features_hdtv_row=[]
    for i in range(0,len(all_features_hdtv[k]),10): #index
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(1,len(all_features_hdtv[k]),10): #reb
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(2,len(all_features_hdtv[k]),10): #bitrate
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(3,len(all_features_hdtv[k]),10): #c_size
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(4,len(all_features_hdtv[k]),10): #width
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(5,len(all_features_hdtv[k]),10): #height
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(6,len(all_features_hdtv[k]),10): #isbest
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(7,len(all_features_hdtv[k]),10): #psnr
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(8,len(all_features_hdtv[k]),10): #ssim
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    for i in range(9,len(all_features_hdtv[k]),10): #vmaf
        all_features_hdtv_row.append(all_features_hdtv[k][i])
    all_features_hdtv_grouped.append(all_features_hdtv_row)
all_features_hdtv_grouped=np.array(all_features_hdtv_grouped)

#split
seed=42
conta = 0

#hdtv
save_mae_importance_for_all_hdtv=[]
save_rmse_importance_for_all_hdtv=[]
maes_test=[]
rmses_test=[]
for each_user_y in users_score_hdtv:
    y_user=each_user_y
    features_train, features_test, y_user_train, y_user_test = train_test_split(all_features_hdtv_grouped, y_user, random_state=seed,test_size=0.3, shuffle=True)
    # normalizza
    # train test
    scaler = MinMaxScaler()
    scaler.fit(features_train)
    X_scaled_train = scaler.transform(features_train)
    X_scaled_test = scaler.transform(features_test)

    reg = svm.SVR(kernel='rbf', gamma=0.5, C=100)

    reg.fit(X_scaled_train, y_user_train)

    p = reg.predict(X_scaled_test)
    mae=mean_absolute_error(y_user_test, p)
    rmse=sqrt(mean_squared_error(y_user_test, p))
    print('user '+str(conta)+' hdtv done')
    print(mae)
    print(rmse)
    maes_test.append(mae)
    rmses_test.append(rmse)

    # Calculate permutation importance for each group
    num_groups = 10
    group_size = 7
    n_repeats = 30
    group_importances = []
    group_importances_sqrt = []
    group_std = []
    group_std_sqrt = []
    for i in range(num_groups):
        importances = []
        importances_sqrt = []
        # Determine the indices for the current group
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_indices = list(range(start_idx, end_idx))
        other_indices = list(range(0, start_idx)) + list(range(end_idx, X_scaled_test.shape[1]))

        for _ in range(n_repeats):
            # Shuffle the group of features together while keeping others intact
            # Create a copy of the feature matrix and shuffle the specific group of features
            X_to_be_shuffled = X_scaled_test.copy()
            # Shuffle the columns indexed by group_indices individually
            for col_idx in group_indices:
                np.random.shuffle(X_to_be_shuffled[:, col_idx])

            # Assign the shuffled group columns back to their original positions
            X_shuffled = X_scaled_test.copy()
            X_shuffled[:, group_indices] = X_to_be_shuffled[:, group_indices]


            # Calculate the importance score after shuffling the group
            shuffled_score = mean_absolute_error(y_user_test,reg.predict(X_shuffled))
            shuffled_score_sqrt = sqrt(mean_squared_error(y_user_test, reg.predict(X_shuffled)))

            # Calculate the importance as the difference in scores
            importance = shuffled_score-mae
            importance_sqr = shuffled_score_sqrt - rmse

            importances.append(importance)
            importances_sqrt.append(importance_sqr)
        # Calculate mean and standard deviation of importance scores
        mean_importance = np.mean(importances)
        std_importance = np.std(importances)
        group_importances.append(mean_importance)
        group_std.append(std_importance)
        mean_importance_sqrt = np.mean(importances_sqrt)
        std_importance_sqrt = np.std(importances_sqrt)
        group_importances_sqrt.append(mean_importance_sqrt)
        group_std_sqrt.append(std_importance_sqrt)

    conta += 1
    save_mae_importance_for_all_hdtv.append(group_importances)
    save_rmse_importance_for_all_hdtv.append(group_importances_sqrt)
np.save('output_data/save_mae_importance_for_all_hdtv.npy',save_mae_importance_for_all_hdtv)
np.save('output_data/save_rmse_importance_for_all_hdtv.npy',save_rmse_importance_for_all_hdtv)