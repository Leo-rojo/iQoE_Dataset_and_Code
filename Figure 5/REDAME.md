## Generate Figure 5
* input files: folder 'users' which contains info of the 120 users that took the assesment, double_users folder which contains the users info that repeated the assessment
* Run generate_dataset.py to generate the well formatted .xls files that contains user's information. The generated files are saved in dataset folder
* Run scores_distribution.py to generate the scores boxplot for atypical users and average user.
* Run time_inconsistencies.py to generate the PLCC and SRCC of the two users that repeated the assessment. The graphs will be stored in double_users/figures folder.
* Run calculate_features_for_iqoe.py to generate feat_iqoe.npy which serves as input to 'read_the_stall_for_each_exp_of_each_user.py' which gives in output tot_viewtime_per_user_50.npy and tot_viewtime_per_user_120.npy
  * Run Times.py to generate the ECDF of the playback and pure viewtimes for all the users.