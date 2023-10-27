## P1203 & LSTM
* LSTM: run generate_features_for_testset_LSTM.py to generate the scores given by LSTM model to the test videos. The output file is scoresbiqps.npy
  * The mentioned file read the experiences that compose the test set and calculate the IFs considered by LSTM model. In particular it also calculate the QP values of chunks. Info about videos 
  are taken from '/Subjective_assesment_iQoE_implementation/video_generations/Video_preparation/mkvfiles' which need to contain the proper files.
* P1203: run generate_features_for_testset_p1203.py to generate the scores given by P1203 model to the test videos. The output file is p1203.npy
* run file mae_rmse_sota_models.py to generate p1203_mae_each_user.npy, p1203_rmse_each_user.npy, lstm_mae_each_user.npy, lstm_rmse_each_user.npy and save them into results_collection folder