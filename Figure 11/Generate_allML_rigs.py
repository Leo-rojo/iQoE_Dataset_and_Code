from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples, pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.base import clone
import pandas as pd
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from modAL.models import ActiveLearner,CommitteeRegressor
import warnings
warnings.filterwarnings("ignore")

# AL strategies
def random_sampling(classifier, X_pool,rng):
    n_samples = len(X_pool)
    query_idx = rng.choice(range(n_samples))
    return query_idx, X_pool[query_idx]
def greedy_sampling_input(regressor, X_pool,rng):
    dist_matrix = pairwise_distances(regressor.X_training, X_pool)
    dist_to_training_set = np.amin(dist_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X_pool[query_idx]
def cluster_uncertainty(regressor, X_pool, n_c=7, n_instances=1):
    query_idx = []
    kmeans = KMeans(n_c)
    y_pool = pd.DataFrame(regressor.predict(X_pool), columns=['y'])
    kmeans.fit(X_pool)
    y_pool['cluster'] = kmeans.labels_
    y_pool['silhouette'] = silhouette_samples(y_pool['y'].to_numpy().reshape(-1, 1), y_pool['cluster'])
    selected_clusters = y_pool.groupby('cluster').agg({'y': 'var'}).nlargest(n_instances, 'y').index.tolist()
    for cluster in selected_clusters:
        query_idx.append(y_pool[y_pool['cluster'] == cluster]['silhouette'].idxmin())
    return query_idx
def random_greedy_sampling_input_output(regressor, X): #it is iGS
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(regressor.X_training, X)
    dist_y_matrix = pairwise_distances(
        regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]
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

def each_user(nr_chunks, rs, u, model, regr_choosen,n_queries):
    rng = np.random.default_rng(42)
    print('start ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))
    # Constants
    nr_feat = nr_chunks * 10

    # all features
    synthetic_experiences = np.load('./features_generated_experiences/feat_iQoE_for_synth_exp.npy')
    scores_synthetic_users = np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')

    X = synthetic_experiences
    y = scores_synthetic_users[u][model].reshape(1000)
    model_name = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][model]

    # define min max scaler
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=rs, shuffle=True)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ###Active_leanring###
    n_initial = 1
    initial_idx = rng.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y_train,dtype=int)[initial_idx]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    Regressors_considered=[RandomForestRegressor(n_estimators = 50, max_depth = 60)]
    Regressors_considered.append(xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1))
    Regressors_considered.append(sklearn.svm.SVR(kernel = 'rbf', gamma= 0.5, C= 100))
    Regressors_considered.append(GaussianProcessRegressor(kernel=RationalQuadratic()+2,alpha=5))

    regr_choosen_idx=['RF', 'XGboost', 'SVR','GP'].index(regr_choosen)
    regr_1 = Regressors_considered[regr_choosen_idx]

    regr_5 = clone(regr_1)
    regressor_gsio = ActiveLearner(
        estimator=regr_5,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )

    ##initial scores
    scores_gsio = [regressor_gsio.score(X_test, y_test)]

    ##initial lcc
    lccs_gsio = [pearsonr(regressor_gsio.predict(X_test), y_test)[0]]

    ##initial srocc
    srccs_gsio = [spearmanr(regressor_gsio.predict(X_test), y_test)[0]]

    ##initial knds
    knds_gsio = [kendalltau(regressor_gsio.predict(X_test), y_test)[0]]

    # initial maes
    maes_gsio = [mean_absolute_error(y_test, regressor_gsio.predict(X_test))]

    #initial rmse
    rmses_gsio = [sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test)))]

    X_pool_gsio=X_pool.copy()
    y_pool_gsio=y_pool.copy()

    # active learning
    t_s=10
    count_queries=1
    for idx in range(n_queries):

        #take random queries
        if count_queries<t_s:
            n_samples = len(X_pool_gsio)
            query_idx = rng.choice(range(n_samples))

        # gsio
        if count_queries >= t_s:
            query_idx, query_instance = regressor_gsio.query(X_pool_gsio)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)



        #save_queries scores
        scores_gsio.append(regressor_gsio.score(X_test, y_test))

        # save_queries lccs
        lccs_gsio.append(pearsonr(regressor_gsio.predict(X_test), y_test)[0])

        # save_queries rmse
        rmses_gsio.append(sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test))))

        ##save_queries srocc
        srccs_gsio.append(spearmanr(regressor_gsio.predict(X_test), y_test)[0])

        ##save_queries knds
        knds_gsio.append(kendalltau(regressor_gsio.predict(X_test), y_test)[0])

        #save_queries maes
        maes_gsio.append(mean_absolute_error(y_test, regressor_gsio.predict(X_test)))

        #print('training_query: '+str(count_queries))
        count_queries+=1

    #salve nelle folder shuffle
    # folders for metrics
    scores100=[scores_gsio]
    lcc100=[lccs_gsio]
    rmse100=[rmses_gsio]
    maes100=[maes_gsio]
    knd100=[knds_gsio]
    srcc100=[srccs_gsio]
    sco=[scores100,lcc100,rmse100,srcc100,maes100,knd100]
    conta=0
    for met in ['R2', 'lcc', 'rmse', 'srcc', 'mae', 'knd']:
        main_path_save = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunks)+'_'+str(n_initial)+'rigs'
        if not os.path.exists(main_path_save + '/' + model_name + '/user_' + str(u) +'/shuffle_'+str(rs)+'/'+met):
            os.makedirs(main_path_save + '/' + model_name + '/user_' + str(u) +'/shuffle_'+str(rs)+'/'+met)
        np.save(main_path_save + '/' + model_name + '/user_' + str(u) + '/shuffle_'+str(rs)+ '/'+met+'/scores_for_ALstrat', sco[conta]) #salvo rigs
        conta+=1
    print('end ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))

if __name__ == "__main__":
    from multiprocessing import Pool
    nr_chunk=7
    n_queries = 250
    # params
    comb_of_par = []

    for reg in ['RF','XGboost','SVR','GP']:
        for rs in [42,13,70,34,104]:
                for u in range(32):
                    for m in range(8):
                        model_name=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][m]
                        main_path =reg + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunk)+'_'+str(1)+'rigs'
                        if not os.path.exists(main_path + '/' + model_name + '/user_' + str(u) + '/shuffle_'+str(rs)):
                            comb_of_par.append((nr_chunk, rs, u, m, reg, n_queries))
                            print(str(nr_chunk)+ '_' + str(rs)+'_' + str(u) +'_'+ str(m))
    print('param missing: '+ str(len(comb_of_par)))

    with Pool() as p:
        #p.map(each_user, [u for u in range(32)])
        p.starmap(each_user, comb_of_par)
    p.close()
    print('done')


