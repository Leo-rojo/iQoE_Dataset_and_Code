## Generate synthetic users, synthetic scores and features for synthetic experiences:
* input data: The folder 'features_and_scores_WIV_hdtv_users/original_waterlooIV_dataset' which contains the original waterloo dataset, the file 'experiences_with_features.npy'.

NB. 'experience_with_features.npy' has been previously created by following the same approach used for real user in iQoE web application. So we runned park simulator with real network traces and considered 306 experiences composed by 30 chunks of 4 secs(2 minutes of original ToS). The file can be found in Subjective_assesment_iQoE_implementation\video_generations\Experience_pool_generation\ABR_simulator\experience_collection

* Run Save_features_and_user_scores_W4.py in 'features_and_scores_WIV_hdtv_users' in order to generate 'all_feat_hdtv.npy' and 'users_scores_hdtv.npy' which contains respectively the features and the scores of real hdtv users of the original dataset. Put the 'users_scores_hdtv.npy' outside the folder.
* Run 'creation_of_features_for_synthetic_users.py' inside folder 'features_for_synthetic_user_fitting'  to generate various files containing different collection of features for the generated videostreaming experience. Each collection is used by different QoE models for generating the synthetic users.
* Run 'fitting_of_models_to_real_users.py' to generate synthetic users that gives unbounded scores, the output files will be saved in Fitted_models_without_logistic
* Run 'Bound_with_logistic' which generate the sigmoid parameters for each synthetich users, so to bound their scores into a scale [1,100]. The output files are stored in 'save_param_sigmoids'
* Run 'Synthetic_users_give_score_to_generated_experiences.py' in order to obtain the scores for the 1000, 7 chunks long generated experiences. The output files produced are: a series of npy files called feat_qoemodel_for_synth_exp.npy containing the features for each different qoemodel used for synthetic user building, the scores given by the synthetic users in 'synthetic_users_scores_for_generated_experiences' and the iQoE features for the generated experiences in 'features_generated_experiences'.

NB. 'Synthetic_users_give_score_to_generated_experiences.py' also select a subset of 1000 experiences composed by 7 chunks of 4 secs each, from the original 306 experiences. 

* in order to create the scores prediction from p1203 and LSTM models for the synthetic users follow the instruction inside the folder 'p1203_and_LSTM_models'