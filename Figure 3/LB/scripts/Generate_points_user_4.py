from sklearn import svm
from modAL.models import ActiveLearner,CommitteeRegressor
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.base import clone
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


rando=42
nr_explorative_queries=0
nq=50
device='hdtv'
folder_figs='../output_data'
if not os.path.exists(folder_figs):
    os.makedirs(folder_figs)

results_all_regressors_mae=[]
results_all_regressors_rmses=[]
results_all_regressors_std_mae=[]
results_all_regressors_std_rmses=[]

#find minimum of bitrates of all devices
all_bits_each_device=[]
#calculate min bitrate of all devices
features_total=np.load('../input_data/features_'+device+'.npy')

# select the main features on which to do the clustering
nr_c=7

all_bits=[]
for exp in features_total:
    # min training bitrate
    bit = []
    for exp in features_total:
        for i in range(2, (2 + nr_c * 13 - 1), 13):
            bit.append(float(exp[i]))
    all_bits.append(bit)
all_bits_each_device.append(all_bits)
min_all=np.min(np.array(all_bits_each_device).flatten())
print('min_all',min_all)

#calculate features for each device
maes_test = []
rmses_test = []
users_scores=np.load('../input_data/users_raters_'+device+'.npy')
features_total=np.load('../input_data/features_'+device+'.npy')

features_selected = []  # sum all [representation_index,rebuffering_duration,video_bitrate,chunk_duration,chunk_size,qp,framerate,width,height,is_best,psnr,ssim,vmaf]
for exp in features_total:
    bit = []
    logbit = []
    for i in range(2, (2 + nr_c * 13 - 1), 13):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_all)
        logbit.append(bit_log)
    # sumbit
    s_bit = np.array(bit).sum()
    # sumlogbit
    l_bit = np.array(logbit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 13 - 1), 13):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    # ave of all reb
    s_reb_ave = np.array(reb).mean()
    # nr of stall
    nr_stall = np.count_nonzero(reb)
    # duration stall+normal
    tot_dur_plus_reb = nr_c * 4 + s_reb

    # psnr
    psnr = []
    for i in range(10, (1 + nr_c * 13 - 1), 13):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    # ssim
    ssim = []
    for i in range(11, (1 + nr_c * 13 - 1), 13):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    # vmaf
    vmaf = []
    for i in range(12, (1 + nr_c * 13 - 1), 13):
        vmaf.append(float(exp[i]))
    # sum
    s_vmaf = np.array(vmaf).sum()
    # ave
    s_vmaf_ave = np.array(vmaf).mean()

    # is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(9, (1 + nr_c * 13 - 1), 13):
        isbest.append(float(exp[i]))

    is_best = np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 4
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([4 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    # collection
    features_selected.append([s_vmaf, s_reb / tot_dur_plus_reb]) ####representation_index,rebuffering_duration,video_bitrate,chunk_duration,chunk_size,qp,framerate,width,height,is_best,psnr,ssim,vmaf

n_feat=len(features_selected[0])
features = np.array(features_selected).reshape(-1, n_feat)

#two ss strategies
def random_sampling(classifier, X_pool,rng):
    n_samples = len(X_pool)
    query_idx = rng.choice(range(n_samples))
    return query_idx, X_pool[query_idx]
def random_greedy_sampling_input_output(regressor, X):  # it is iGS
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(regressor.X_training, X)
    dist_y_matrix = pairwise_distances(
        regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]
def greedy_sampling_input(regressor, X_pool):
    dist_matrix = pairwise_distances(regressor.X_training, X_pool)
    dist_to_training_set = np.amin(dist_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X_pool[query_idx]
def cluster_uncertainty(regressor, X_pool, n_c=3, n_instances=1):
    query_idx = []
    kmeans = KMeans(n_c,random_state=0)
    y_pool = pd.DataFrame(regressor.predict(X_pool), columns=['y'])
    kmeans.fit(X_pool)
    y_pool['cluster'] = kmeans.labels_
    y_pool['silhouette'] = silhouette_samples(y_pool['y'].to_numpy().reshape(-1, 1), y_pool['cluster'])
    selected_clusters = y_pool.groupby('cluster').agg({'y': 'var'}).nlargest(n_instances, 'y').index.tolist()
    for cluster in selected_clusters:
        query_idx.append(y_pool[y_pool['cluster'] == cluster]['silhouette'].idxmin())
    return query_idx
def committ_qs(committee, X_pool):  # Pool-Based Sequential Active Learning for Regression Dongrui Wu, qbc with variance taken from other paper
    # print(X_pool)
    variances = []
    vote_learners = committee.vote(X_pool)  # obtain prediction of various learners
    for i in vote_learners:
        mean = np.mean(i)
        s = 0
        for k in range(len(committee)):
            s += (i[k] - mean) ** 2
        s = s / len(committee)
        variances.append(s)
    # print(variances)
    query_idx = np.argmax(variances)
    return query_idx, X_pool[query_idx]

random_32_users_mae=[]
igs_32_users_mae=[]
random_32_users_rmse = []
igs_32_users_rmse = []
uc_32_users_mae=[]
qbc_32_users_mae=[]
gs_32_users_mae=[]
uc_32_users_rmse = []
qbc_32_users_rmse = []
gs_32_users_rmse = []
save_queries_random=[]
save_queries_igs=[]
save_queries_uc=[]
save_queries_qbc=[]
save_queries_gs=[]

#iqoe with random and with riGS for each user
save_y_train=[]
for user in [4]:
    save_queries_random_user = []
    save_queries_igs_user = []
    save_queries_uc_user = []
    save_queries_qbc_user = []
    save_queries_gs_user = []
    save_y_random_user = []
    save_y_igs_user = []
    save_y_uc_user = []
    save_y_qbc_user = []
    save_y_gs_user = []
    rs = 42
    rng = np.random.default_rng(rando)
    print('start user ' + str(user))
    # Constants
    nr_feat = 2

    X = features
    y = users_scores[user]  # .reshape(450)

    # define min max scaler
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs, shuffle=True)
    save_y_train.append(y_train)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    n_queries = nq#len(X_train)-1

    ###Active_leanring###
    n_initial = 1
    initial_idx = rng.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y, dtype=int)[initial_idx]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    regr_1 = sklearn.svm.SVR(kernel='rbf', gamma=0.5, C=100)

    regressor_random = ActiveLearner(
        estimator=regr_1,
        query_strategy=random_sampling,
        X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
    )

    regr_2 = clone(regr_1)
    regressor_gsio = ActiveLearner(
        estimator=regr_2,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )

    regr_3 = clone(regr_1)
    regressor_cluster = ActiveLearner(
        estimator=regr_3,
        query_strategy=cluster_uncertainty,
        X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
    )

    regr_4 = clone(regr_1)
    regressor_gs = ActiveLearner(
        estimator=regr_4,
        query_strategy=greedy_sampling_input,
        X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
    )

    # initializing Committee members
    n_members = 3
    learner_list = list()
    # a list of ActiveLearners:
    for member_idx in range(n_members):
        # initializing learner
        learner = ActiveLearner(
            estimator=clone(regr_1),
            X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
        )
        learner_list.append(learner)
    # inp output greedy sampling
    regressor_comm = CommitteeRegressor(learner_list=learner_list, query_strategy=committ_qs)


    # initial maes
    maes_r = [mean_absolute_error(y_test, regressor_random.predict(X_test))]
    maes_gsio = [mean_absolute_error(y_test, regressor_gsio.predict(X_test))]
    maes_uc = [mean_absolute_error(y_test, regressor_cluster.predict(X_test))]
    maes_gs = [mean_absolute_error(y_test, regressor_gs.predict(X_test))]
    maes_comm = [mean_absolute_error(y_test, regressor_comm.predict(X_test))]


    # initial rmse
    rmses_r = [sqrt(mean_squared_error(y_test, regressor_random.predict(X_test)))]
    rmses_gsio = [sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test)))]
    rmses_uc = [sqrt(mean_squared_error(y_test, regressor_cluster.predict(X_test)))]
    rmses_gs = [sqrt(mean_squared_error(y_test, regressor_gs.predict(X_test)))]
    rmses_comm = [sqrt(mean_squared_error(y_test, regressor_comm.predict(X_test)))]

    X_pool_random,X_pool_gsio,X_pool_cluster,X_pool_comm,X_pool_gs = X_pool.copy(), X_pool.copy(), X_pool.copy(), X_pool.copy(), X_pool.copy()
    y_pool_random, y_pool_gsio,y_pool_cluster,y_pool_comm,y_pool_gs = y_pool.copy(), y_pool.copy(), y_pool.copy(), y_pool.copy(), y_pool.copy()

    # active learning
    t_s = nr_explorative_queries
    count_queries = 1
    for idx in range(n_queries):

        # take random queries
        if count_queries < t_s:
            n_samples = len(X_pool_random)
            query_idx = rng.choice(range(n_samples))
            #print('common_random_qidx: '+str(query_idx))
        # random
        if count_queries >= t_s:
            query_idx, query_instance = regressor_random.query(X_pool_random,rng)
            #print('random_ss_qidx: ' + str(query_idx))
        query_idx = int(query_idx)
        save_queries_random_user.append(X_pool_random[query_idx])
        save_y_random_user.append(y_pool_random[query_idx])
        regressor_random.teach(np.array(X_pool_random[query_idx]).reshape(-1, nr_feat),
                               np.array(y_pool_random[query_idx]).reshape(-1, 1).flatten())
        X_pool_random, y_pool_random = np.delete(X_pool_random, query_idx, axis=0), np.delete(y_pool_random, query_idx)

        # gsio
        if count_queries >= t_s:
            query_idx, query_instance = regressor_gsio.query(X_pool_gsio)
            #print('igs_ss_qidx: ' + str(query_idx))
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        save_queries_igs_user.append(X_pool_gsio[query_idx])
        save_y_igs_user.append(y_pool_gsio[query_idx])
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                             np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)

        # cluster
        if count_queries >= t_s:
            query_idx, query_instance = regressor_cluster.query(X_pool_cluster, n_c=7, n_instances=1)
            query_idx = int(query_idx[0])
        else:
            query_idx = int(query_idx)
        save_queries_uc_user.append(X_pool_cluster[query_idx])
        regressor_cluster.teach(np.array(X_pool_cluster[query_idx]).reshape(-1, nr_feat),
                                np.array(y_pool_cluster[query_idx]).reshape(-1, 1).flatten())
        X_pool_cluster, y_pool_cluster = np.delete(X_pool_cluster, query_idx, axis=0), np.delete(y_pool_cluster, query_idx)

        # gs
        if count_queries >= t_s:
            query_idx, query_instance = regressor_gs.query(X_pool_gs)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        save_queries_gs_user.append(X_pool_gs[query_idx])
        save_y_gs_user.append(y_pool_gs[query_idx])
        regressor_gs.teach(np.array(X_pool_gs[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_gs[query_idx]).reshape(-1, 1).flatten())
        X_pool_gs, y_pool_gs = np.delete(X_pool_gs, query_idx, axis=0), np.delete(y_pool_gs, query_idx)

        # committee
        if count_queries >= t_s:
            query_idx, query_instance = regressor_comm.query(X_pool_comm)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)
        save_queries_qbc_user.append(X_pool_comm[query_idx])
        regressor_comm.teach(np.array(X_pool_comm[query_idx]).reshape(-1, nr_feat),
                             np.array(y_pool_comm[query_idx]).reshape(-1, 1).flatten())
        X_pool_comm, y_pool_comm = np.delete(X_pool_comm, query_idx, axis=0), np.delete(y_pool_comm, query_idx)


        # save_queries rmse
        rmses_r.append(sqrt(mean_squared_error(y_test, regressor_random.predict(X_test))))
        rmses_gsio.append(sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test))))
        rmses_uc.append(sqrt(mean_squared_error(y_test, regressor_cluster.predict(X_test))))
        rmses_gs.append(sqrt(mean_squared_error(y_test, regressor_gs.predict(X_test))))
        rmses_comm.append(sqrt(mean_squared_error(y_test, regressor_comm.predict(X_test))))

        # save_queries maes
        maes_r.append(mean_absolute_error(y_test, regressor_random.predict(X_test)))
        maes_gsio.append(mean_absolute_error(y_test, regressor_gsio.predict(X_test)))
        maes_uc.append(mean_absolute_error(y_test, regressor_cluster.predict(X_test)))
        maes_gs.append(mean_absolute_error(y_test, regressor_gs.predict(X_test)))
        maes_comm.append(mean_absolute_error(y_test, regressor_comm.predict(X_test)))

        count_queries += 1
    save_queries_random.append(save_queries_random_user)
    save_queries_igs.append(save_queries_igs_user)
    save_queries_uc.append(save_queries_uc_user)
    save_queries_gs.append(save_queries_gs_user)
    save_queries_qbc.append(save_queries_qbc_user)
    np.save(folder_figs+'/save_queries_random_'+str(user), save_queries_random)
    np.save(folder_figs+'/save_queries_igs_'+str(user), save_queries_igs)
    np.save(folder_figs+'/save_queries_uc_'+str(user), save_queries_uc)
    np.save(folder_figs+'/save_queries_gs_'+str(user), save_queries_gs)
    np.save(folder_figs+'/save_queries_qbc_'+str(user), save_queries_qbc)
    np.save(folder_figs+'/save_y_random_'+str(user), save_y_random_user)
    np.save(folder_figs+'/save_y_igs_'+str(user), save_y_igs_user)
    np.save(folder_figs+'/save_y_uc_'+str(user), save_y_uc_user)
    np.save(folder_figs+'/save_y_gs_'+str(user), save_y_gs_user)
    np.save(folder_figs+'/save_y_qbc_'+str(user), save_y_qbc_user)


    random_32_users_mae.append(maes_r)
    igs_32_users_mae.append(maes_gsio)
    random_32_users_rmse.append(rmses_r)
    igs_32_users_rmse.append(rmses_gsio)
    uc_32_users_mae.append(maes_uc)
    gs_32_users_mae.append(maes_gs)
    qbc_32_users_mae.append(maes_comm)
    uc_32_users_rmse.append(rmses_uc)
    gs_32_users_rmse.append(rmses_gs)
    qbc_32_users_rmse.append(rmses_comm)

name=[]
for k in range(32):
    name.append('user'+'_'+str(k))

#save good user
good_users=[]
for i in range(len(random_32_users_mae)):
    igs=igs_32_users_mae[i][-1]
    r=random_32_users_mae[i][-1]
    uc=uc_32_users_mae[i][-1]
    gs=gs_32_users_mae[i][-1]
    qbc=qbc_32_users_mae[i][-1]
    if igs<r and igs<uc and igs<gs and igs<qbc:
        good_users.append(i)
        print(i)
        print(name[i])
        print([r-igs, uc-igs, gs-igs, qbc-igs])
        print('--------')

name_two_feat=['s_vmaf','s_reb/tot_dur+reb']
font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 60}
plt.rc('font', **font_general)

for user in good_users:
    #graph all features
    traj_random=save_queries_random[user]
    traj_igs=save_queries_igs[user]
    traj_uc=save_queries_uc[user]
    traj_gs=save_queries_gs[user]
    traj_qbc=save_queries_qbc[user]
    ss=['random','igs','uc','gs','qbc']
    fig = plt.figure(figsize=(30, 10), dpi=100)
    x = X_train[:, 0]
    y = X_train[:, 1]
    np.save(folder_figs+'/all_space_points_user_4.npy',[x,y])
    np.save(folder_figs+'/y_train_user_4.npy',save_y_train[user])