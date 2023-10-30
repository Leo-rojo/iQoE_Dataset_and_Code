## Generate p1203 and LSTM scores predictions for synthetic users:
#### input files: 
* feat_iQoE_for_synth_exp.npy, file generated in subjective_assessments 
* chunk_features_not_qp folder which contains IFs for LSTM model wihtout qp values for the 30 chunks for different encodings. (we do not provide the code for its generation, but we put the folder already done) 
* encoded_video_chunked which contains the actual encoded chunks for calculation of QP values (LSTM model needs the QP values of the actual videos). 
For memory reason copy of this folder can be downloaded from https://doi.org/10.6084/m9.figshare.24460078 in subfolder Simulation_chunks.
* exp_bb.npy, exp_mpc.npy, exp_th.npy generated in subjective_assessments
#### Script to be run:
* Run "p1203_scores.py" to generate the scores given by p1203 linearly scaled between 1-100 for all the 1000 experiences, 
the results will be saved in file "p1203_scores.npy" which is used for figure 16.
* Run "LSTM_scores.py" to generate the scores given by LSTM model for all the 1000 experiences, 
the results will be saved in file "scoresbiqps.npy" which is used for figure 16. In order to run it you need to install the LSTM model, 
link: https://github.com/acmmmsys/2020-BiQPS

#### Output files:
* p1203_scores.npy, file generated in p1203_scores.npy
* scoresbiqps.npy, file generated in scoresbiqps.npy
* chunk_features_qp which is just a folder for intermidiate results of LSTM_scores.py (same of chunk_features_not_qp but with qp IFs calculated for each chunks)


N.B. LSTM_scores.py uses "ffmpeg-debug-qp" (https://github.com/slhck/ffmpeg-debug-qp) to caclulate qp values of videos. So it need to be installed.