## Generate synthetic users, synthetic scores and features for synthetic experiences:

* Input data: The folder 'features_and_scores_WIV_hdtv_users/original_waterlooIV_dataset' which contains the original waterloo dataset, 
the file 'experiences_with_features.npy' which can be generated in Subjective_assessments, which consists of 306 experiences composed by 30 chunks of 4 secs(2 minutes of original ToS).
* Run generate_all_files.py to create all the necessary files in the "output_data" directory for conducting experiments with synthetic users. The file run the following scripts in folder script:
  * Run Save_features_and_user_scores_W4.py 
  * Run 'creation_of_features_for_synthetic_users.py' 
  * Run 'fitting_of_models_to_real_users.py' 
  * Run 'Bound_with_logistic' 
  * Run 'Synthetic_users_give_score_to_generated_experiences.py' which consists of a subset of 1000 experiences composed by 7 chunks of 4 secs each, from the original 306 experiences.

p1203_scores.npy and scoresbiqps.npy are copied to output_folder from folder p1203_and_LSTM_models. In order to create the scores prediction from p1203 and LSTM models for the synthetic users follow the instruction inside the folder 'p1203_and_LSTM_models'