## Generate Figure 6

### LA
* Input data: mae and rmse results for each users for the 8 baselines, iQoE, and p1203 and LSTM models as .npy files
* Run generate_figure_6.py to generate figure 6
* Run Plot_legend_dashed.py to generate the legend for figure 6

### LB
* Input data: folder 'users' which contains info of the 120 users that took the assesment, double_users folder which contains the users info that repeated the assessment
* Run generate_figure_6.py to generate figure 6 and run all the other scripts contained in scripts. Those files produce in output folder:
  * mae and rmse results for each users for the 8 baselines, iQoE, and p1203 and LSTM models as .npy files
  * features for the 8 QoE models B, G, V, R, S, N, F, and A as .npy files in the folder 'features_qoes_test' and 'features_qoes_train'
  * files params_mos.npy contaning the parameters of the trained models and videoAtlas_mos.pkl containing the trained videoatlas model
  * some .xls files containing supporting information

### LSTM and P1203
in /scripts/sotamodels/Test_videos_iqoe are present various scripts and files to calculate the mae and rmse for the LSTM and P1203 models on the test dataset. We don't have access to
the training modules of those models so we take their implementation as it is. In order to calculate the features for the LSTM model and P1203 the scripts are:
* generate_features_for_testset_lstm.py which need the path to the video files used in the subjective assessment and the path to the ffmpeg_debug_qp for calcualting qp of video chunks. We already provided the results of this scripts.
* generate_features_for_testset_p1203.py, we already provided the results of this scripts. 
All datas and results related to LSTM and P1203 are contained in the folder /scripts/sotamodels/Test_videos_iqoe