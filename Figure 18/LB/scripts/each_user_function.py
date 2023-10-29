from sklearn.model_selection import train_test_split
import numpy as np
from modAL.models import ActiveLearner
from sklearn.metrics import pairwise_distances
import copy
from sklearn.preprocessing import MinMaxScaler
import sklearn
import warnings
import pickle
import time
warnings.filterwarnings("ignore")
# AL strategies
def random_greedy_sampling_input_output(regressor, X, switch,rng):
    if not switch:
        n_samples = len(X)
        query_idx = rng.choice(range(n_samples))
    else:
        y = regressor.predict(X)
        dist_x_matrix = pairwise_distances(regressor.X_training, X)
        dist_y_matrix = pairwise_distances(
            regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
        )
        dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
        query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]
def each_user(nr_chunks, rs, u, model, regr_choosen,n_queries):
    rng = np.random.default_rng(42)
    save_time_queries=[]
    print('start ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))
    # Constants
    nr_feat = nr_chunks * 10

    # all features
    synthetic_experiences = np.load('../input_data/features_generated_experiences/feat_iQoE_for_synth_exp.npy')
    scores_synthetic_users = np.load('../input_data/synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')

    all_features = copy.copy(synthetic_experiences)
    users_scores = copy.copy(scores_synthetic_users)

    X = all_features
    y = scores_synthetic_users[u][model].reshape(1000)
    model_name = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][model]

    # define min max scaler
    scaler = MinMaxScaler()


    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=rs,shuffle=True)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ###Active_leanring###
    n_initial = 1
    regr_5 = sklearn.svm.SVR(kernel = 'rbf', gamma= 0.5, C= 100)
    start = time.time()
    initial_idx = rng.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y_train,dtype=int)[initial_idx]
    # Isolate the non-training examples we'll be querying.
    regressor_gsio = ActiveLearner(
        estimator=regr_5,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )
    end = time.time()
    pkl_filename = '../output_data/mq/'+str(u)+model_name+'m_q' + 'initial'+'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(regressor_gsio.estimator, file)
    #regressor_gsio.estimator.save_model('./mq/'+str(u)+model_name+'m_q' + 'initial'+'.json')
    save_time_queries.append(end - start)
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)
    X_pool_gsio=X_pool.copy()
    y_pool_gsio=y_pool.copy()


    # active learning
    h = 10
    count_queries = 1
    switch_bol = False
    for idx in range(n_queries):
        if count_queries > h:
            switch_bol = True
        start=time.time()
        # gsio
        query_idx, query_instance = regressor_gsio.query(X_pool_gsio, switch_bol,rng)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)
        end=time.time()
        save_time_queries.append(end-start)
        pkl_filename = '../output_data/mq/'+str(u)+model_name+'m_q' + str(idx)+'.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(regressor_gsio.estimator, file)
        count_queries+=1
    #salve nelle folder shuffle
    # folders for metrics
    print('end ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))
    np.save('../output_data/time_over/time_overhead_'+str(u)+model_name,save_time_queries)