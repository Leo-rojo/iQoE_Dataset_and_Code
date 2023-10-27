import random
import uuid
from flask import Flask, render_template, request, session, make_response
from flask_cors import CORS
from modAL.models import ActiveLearner
import numpy as np
from sklearn.metrics import pairwise_distances
import xgboost as xgb
import sklearn
import shutil
import os
import time
from os.path import exists
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import pickle

application = Flask(__name__,static_folder='cssandjsandvideos')
application.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'
CORS(application)
auth = HTTPBasicAuth()
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
#log = logging.getLogger('werkzeug')
#log.disabled = True

users = {
    "anonym": generate_password_hash("iQoE_92")
}

here = os.path.dirname(__file__)
maps = list([str(i) for i in [6, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 7, 8]])  # from lowest quality to highest. I found this order by analyzing the encoded files with sorth_themkvfiles_with_array
reference_list=[int(i.split('_')[0]) for i in os.listdir(here+'/cssandjsandvideos') if i.split('_')[-1]=='ref.mp4']
nr_chunks=4
nr_feat = nr_chunks * 10
n_queries_total=120 #10 ts + 40 base + 40 iGS + 30 test
n_baselines_queries=40
n_test_queries=30
n_train_queries=int(n_queries_total-n_test_queries)
n_initial = 1
lastmodel=1
train_test_order=np.load(here+'/original_database/train_test_order.npy')
t_s = 10
conta_first_time=0
conta_first_time_train=0
conta_first_time_baseline=0

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username

def iGS(regressor,X_train,Y_train, X): #it is iGS
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(np.array(X_train).reshape(-1, nr_feat), X)
    dist_y_matrix = pairwise_distances(np.array(Y_train).reshape(-1, 1), y.reshape(-1, 1))
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]

#rules_page
@application.route("/",methods=['POST','GET'])
@auth.login_required
def first_page():
    session.clear()
    return render_template('Rules_of_the_assesment.html')

#initial form to ask for reference
@application.route("/reference_video_form",methods=['POST','GET'])
@auth.login_required
def reference_video_form():
    session['first_time']=True
    return render_template('reference_video_form.html')

@application.route("/start",methods=['POST','GET'])
@auth.login_required
def start():
    if session.get('first_time',None):
        session['first_time']=False
        #session['baselines_queries']=False
        identifier = str(uuid.uuid4())
        session['identifier'] = str(identifier)
        user_folder = here+'/user_' + str(identifier)
        shutil.copytree(here+'/original_database', user_folder)
        if not exists(user_folder + '/save_time.txt'):
            f = open(user_folder + '/save_time.txt', "a")
            f.write('start_reference_' + str(session.get('time_reference',None)) + '\n')
            f.write('start_' + str(time.time()) + '\n')
            f.close()
        open(user_folder + '/Scores_' + str(identifier) + '.txt', 'a').close()
        open(user_folder + '/Scores_test_' + str(identifier) + '.txt', 'a').close()
        open(user_folder + '/check_evolution_' + str(identifier) + '.txt', 'a').close()
        open(user_folder + '/Scores_total_' + str(identifier) + '.txt', 'a').close()
        with open(user_folder+'/rng.pkl', 'wb') as outp:
            rng = np.random.default_rng(42)
            pickle.dump(rng, outp, pickle.HIGHEST_PROTOCOL)
        np.save(user_folder + '/baselines_queries', False)
        np.save(user_folder + '/conta', conta_first_time)
        np.save(user_folder + '/conta_train', conta_first_time_train)
        np.save(user_folder + '/conta_baseline', conta_first_time_baseline)
        X_pool_igs_train = []
        np.save(user_folder + '/X_pool_igs_train', X_pool_igs_train)
        Y_pool_igs_train = []
        np.save(user_folder + '/Y_pool_igs_train', Y_pool_igs_train)


        os.makedirs(user_folder + '/models_' + str(identifier))
        res = make_response(render_template('start.html', query_nr=conta_first_time, total_queries=n_queries_total, max=None, min=None,average=None, previous=None))  ####insert html with video tag here
        res.set_cookie('identifier', str(identifier),max_age=60*60*24*5) #5days
        return res
    else:
        return render_template('start.html', query_nr=conta_first_time, total_queries=n_queries_total, max=None, min=None, average=None, previous=None)

@application.route("/start_loop",methods=['POST','GET'])
@auth.login_required
def elaborations_after_score():
    identifier=session.get('identifier',None)
    user_folder = here+'/user_' + str(identifier)
    with open(user_folder+'/rng.pkl', 'rb') as inp:
        rng = pickle.load(inp)
    #log
    f = open(user_folder + '/save_time.txt', "a")
    f.write('score_given_' + str(time.time()) + '\n')
    f.close()
    # take score of preceiding query
    score = request.form['var1']
    # load counts
    conta = int(np.load(user_folder + '/conta.npy'))
    conta_train = int(np.load(user_folder + '/conta_train.npy'))
    conta_baseline = int(np.load(user_folder + '/conta_baseline.npy'))

    # do calculation
    if np.load(user_folder + '/baselines_queries.npy'):
        # take old
        query_idx_old_baseline = int(np.load(user_folder + '/idx_baseline.npy'))
        idx_col_train = np.load(user_folder + '/idx_col_train.npy')
        idx_col_test = np.load(user_folder + '/idx_col_test.npy')
        all_baselines_queries=np.load(user_folder + '/all_baselines_queries.npy')
        # save score

        file3 = open(user_folder + '/Scores_baseline' + str(identifier) + '.txt', "a")
        file2 = open(user_folder + '/Scores_total_' + str(identifier) + '.txt', "a")
        file3.write('score ' + score + '\n')
        file3.write('video nr ' + str(idx_col_train[query_idx_old_baseline]) + '\n')
        file2.write('score ' + score + '\n')
        file3.close()
        file2.close()
        # select new
        conta_baseline+=1
        np.save(user_folder + '/conta_baseline', conta_baseline)
        conta += 1
        np.save(user_folder + '/conta', conta)

        if conta_baseline==n_baselines_queries:
            np.save(user_folder + '/baselines_queries', False)
            print('baselines_finished')
        else:
            new_idx_baseline = all_baselines_queries[conta_baseline]
            np.save(user_folder + '/idx_baseline', new_idx_baseline)
            print('conta_train=' + str(conta_train))
            print('conta_test=' + str(conta-conta_train-conta_baseline))
            print('conta_baseline=' + str(conta_baseline))
            print('len_train=' + str(len(idx_col_train)))
            print('len_test=' + str(len(idx_col_test)))
            print('still_baseline')
    elif train_test_order[conta-conta_baseline]: #conta-conta_baseline is conta_test+conta_train
        #load old data
        idx_col_train = np.load(user_folder + '/idx_col_train.npy')
        query_idx_old_train = int(np.load(user_folder + '/idx_train.npy'))
        X_train_scaled = np.load(user_folder + '/X_train_scaled.npy')
        synth_exp_train = np.load(user_folder + '/synth_exp_train.npy')
        # save score and video_idx old query
        file1 = open(user_folder + '/Scores_' + str(identifier) + '.txt', "a")
        file2 = open(user_folder + '/Scores_total_' + str(identifier) + '.txt', "a")
        file1.write('score ' + score + '\n')
        file2.write('score ' + score + '\n')
        file1.write('video nr ' + str(idx_col_train[query_idx_old_train]) + '\n')
        file1.close()
        file2.close()
        # model ex novo vs loadmodel
        if len(os.listdir(user_folder + '/models_' + str(identifier))) == 0:
            #reg = xgb.XGBRegressor(n_estimators=100, max_depth=60, nthread=1)
            reg = sklearn.svm.SVR(kernel='rbf', gamma=0.5, C=100)
            lastmodel = 0
        else:
            all_m = [int(lastmodel.split('.')[0]) for lastmodel in os.listdir(user_folder + '/models_' + str(identifier))]
            lastmodel = sorted(all_m)[-1]
            lm = user_folder + '/models_' + str(identifier) + '/' + str(lastmodel) + '.pkl'
            with open(lm, 'rb') as file:
                reg = pickle.load(file)
            #reg = xgb.XGBRegressor()
            #reg.load_model(lm)
        actual_model = lastmodel + 1
        y_score = int(score)  # score captured from user

        #load old training pool
        X_pool_igs_train = np.load(user_folder + '/X_pool_igs_train.npy').tolist()
        Y_pool_igs_train = np.load(user_folder + '/Y_pool_igs_train.npy').tolist()
        # update pool di training
        X_pool_igs_train.append(X_train_scaled[query_idx_old_train])
        Y_pool_igs_train.append(y_score)
        #salva updated trainig pool
        np.save(user_folder + '/X_pool_igs_train', X_pool_igs_train)
        np.save(user_folder + '/Y_pool_igs_train', Y_pool_igs_train)
        #fit model on pool di training
        reg.fit(np.array(X_pool_igs_train).reshape(-1, nr_feat),np.array(Y_pool_igs_train).reshape(-1, 1).flatten())
        #reg.save_model(user_folder + '/models_' + str(identifier) + '/' + str(actual_model) + '.json')
        ####save model######
        pkl_filename = user_folder + '/models_' + str(identifier) + '/' + str(actual_model)+'.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(reg, file)

        ####################check videos#######################
        file1 = open(user_folder + '/check_evolution' + '_' + str(identifier) + '.txt', "a")
        file1.write('query_origi ' + str(query_idx_old_train) + '\n')
        file1.write('query_idx ' + str(idx_col_train[query_idx_old_train]) + '\n')
        file1.close()
        ############################################
        conta += 1
        np.save(user_folder + '/conta', conta)
        conta_train += 1
        np.save(user_folder + '/conta_train', conta_train)
        # update database and save it
        file1 = open(user_folder + '/check_evolution' + '_' + str(identifier) + '.txt', "a")
        file1.write('deleted_scaled ' + str(X_train_scaled[query_idx_old_train]) + '\n')
        file1.write('deleted_not_scaled ' + str(synth_exp_train[query_idx_old_train]) + '\n')
        file1.write('deleted_idx_col ' + str(idx_col_train[query_idx_old_train]) + '\n')
        X_train_scaled = np.delete(X_train_scaled, query_idx_old_train, axis=0)
        synth_exp_train = np.delete(synth_exp_train, query_idx_old_train, axis=0)
        idx_col_train = np.delete(idx_col_train, query_idx_old_train, axis=0)
        np.save(user_folder + '/X_train_scaled', X_train_scaled)
        np.save(user_folder + '/synth_exp_train', synth_exp_train)
        np.save(user_folder + '/idx_col_train', idx_col_train)

        # check if you reached the end of the experiment
        if conta == n_queries_total:
            f = open(user_folder + '/save_time.txt', "a")
            f.write('pause_' + str(time.time()) + '\n')
            f.close()
            print('finish in train')
            return render_template('finish.html')
        ##########################################################################################################one train finish
        if conta_train != n_train_queries:
            X_test_scaled = np.load(user_folder + '/X_test_scaled.npy')
            print('conta_train=' + str(conta_train))
            print('conta_test=' + str(conta - conta_train-conta_baseline))
            print('conta_baseline=' + str(conta_baseline))
            print('len_train=' + str(len(idx_col_train)))
            print('len_test=' + str(len(X_test_scaled)))
            # select next query
            if conta_train < t_s:
                new_idx_train = rng.choice(range(len(X_train_scaled)))
                print('still random')
                file1 = open(user_folder + '/Scores' + '_' + str(identifier) + '.txt', "a")
                #igsornot = 'random_query'
                #file1.write('r_or_iGS ' + igsornot + '\n')
                file1.close()
                new_idx_train = int(new_idx_train)  # 0because it is a list in this particular case
                np.save(user_folder + '/idx_train', new_idx_train)
            elif conta_baseline==0:
                np.save(user_folder + '/baselines_queries', True)
                all_baselines_queries=rng.choice(range(len(X_train_scaled)), n_baselines_queries, replace=False)
                np.save(user_folder + '/all_baselines_queries', all_baselines_queries)
                np.save(user_folder + '/idx_baseline',all_baselines_queries[conta_baseline])
                print('first baselines random')
                file1 = open(user_folder + '/Scores_baseline' + str(identifier) + '.txt', "a")
                file1.close()
            else:  # iGS
                new_idx_train, query_instance = iGS(reg,X_pool_igs_train,Y_pool_igs_train, X_train_scaled)
                print('iGS choosen')
                file1 = open(user_folder + '/Scores' + '_' + str(identifier) + '.txt', "a")
                #igsornot = 'iGS_query'
                #file1.write('r_or_iGS ' + igsornot + '\n')
                file1.close()
                new_idx_train = int(new_idx_train)  # 0because it is a list in this particular case
                np.save(user_folder + '/idx_train', new_idx_train)
    else:
        #load old
        idx_col_test = np.load(user_folder + '/idx_col_test.npy')
        query_idx_old_test = np.load(user_folder + '/idx_test.npy')
        X_test_scaled = np.load(user_folder + '/X_test_scaled.npy')
        synth_exp_test = np.load(user_folder + '/synth_exp_test.npy')
        # save score and video_idx old query
        file1 = open(user_folder + '/Scores_test' + '_' + str(identifier) + '.txt', "a")
        file2 = open(user_folder + '/Scores_total' + '_' + str(identifier) + '.txt', "a")
        file1.write('score ' + score + '\n')
        file2.write('score ' + score + '\n')
        file1.write('video nr ' + str(idx_col_test[query_idx_old_test]) + '\n')
        file1.close()
        file2.close()

        # update database test and save it
        X_test_scaled = np.delete(X_test_scaled, query_idx_old_test, axis=0)
        synth_exp_test = np.delete(synth_exp_test, query_idx_old_test, axis=0)
        idx_col_test = np.delete(idx_col_test, query_idx_old_test, axis=0)
        np.save(user_folder + '/X_test_scaled', X_test_scaled)
        np.save(user_folder + '/synth_exp_test', synth_exp_test)
        np.save(user_folder + '/idx_col_test', idx_col_test)

        conta += 1
        np.save(user_folder + '/conta', conta)

        ##########check if it is finished
        if conta == n_queries_total:
            f = open(user_folder + '/save_time.txt', "a")
            f.write('pause_' + str(time.time()) + '\n')
            f.close()
            return render_template('finish.html')
        ##########################################################################################################one train finish
        conta_test=conta-conta_train-conta_baseline
        if conta_test != n_test_queries:
            # select next query
            X_train_scaled = np.load(user_folder + '/X_train_scaled.npy')
            new_idx_test = rng.choice(range(len(X_test_scaled)))
            np.save(user_folder + '/idx_test', new_idx_test)
            print('conta_train=' + str(conta_train))
            print('conta_test=' + str(conta - conta_train-conta_baseline))
            print('conta_baseline=' + str(conta_baseline))
            print('len_train=' + str(len(X_train_scaled)))
            print('len_test=' + str(len(X_test_scaled)))
            print('still test')

    # calcola le statistiche delle precedenti queries
    with open(user_folder + '/Scores_total_' + str(identifier) + '.txt') as file_in:
        lines = []
        for line in file_in:
            lines.append(int(line.split(' ')[-1]))

    #save random state
    with open(user_folder + '/rng.pkl', 'wb') as outp:
        pickle.dump(rng, outp, pickle.HIGHEST_PROTOCOL)
    return render_template('start_loop.html', query_nr=conta, total_queries=n_queries_total, max=np.max(lines), min=np.min(lines), average=np.mean(lines), previous=lines)

@application.route("/restart",methods=['POST','GET'])
@auth.login_required
def restart():
    #prendi cookie
    identifier=str(request.cookies.get('identifier'))
    session['identifier']=identifier
    if identifier=='None':
        id_not_found='the identifier does not exist'
        return render_template('Rules_of_the_assesment.html',id_not_found=id_not_found)
    else:
        session['first_time'] = False
        user_folder = here+'/user_' + str(identifier)

        #load random state
        with open(user_folder + '/rng.pkl', 'rb') as inp:
            rng = pickle.load(inp)
        conta = int(np.load(user_folder + '/conta.npy'))
        conta_baseline = int(np.load(user_folder + '/conta_baseline.npy'))
        if conta==0:
            f = open(user_folder + '/save_time.txt', "a")
            f.write('restart_but_first_time_' + str(time.time()) + '\n')
            f.close()
            return render_template('start.html', query_nr=conta_first_time, total_queries=n_queries_total, max=None,min=None, average=None, previous=None)
        else:
            id_found='the identifier does exist'
            # save log
            f = open(user_folder + '/save_time.txt', "a")
            f.write('restart_' + str(time.time()) + '\n')
            f.close()
            if np.load(user_folder + '/baselines_queries.npy'):
                idx_col_train = np.load(user_folder + '/idx_col_train.npy')
                query_idx_old_baseline = int(np.load(user_folder + '/idx_baseline.npy'))
                final_video=str(idx_col_train[query_idx_old_baseline]) + '.mp4'
            elif train_test_order[conta-conta_baseline]: #true=training
                idx_col_train = np.load(user_folder + '/idx_col_train.npy')
                query_idx_old_train = int(np.load(user_folder + '/idx_train.npy'))
                final_video=str(idx_col_train[query_idx_old_train]) + '.mp4'
            else:
                idx_col_test = np.load(user_folder + '/idx_col_test.npy')
                query_idx_old_test = int(np.load(user_folder + '/idx_test.npy'))
                final_video=str(idx_col_test[query_idx_old_test]) + '.mp4'
            session['final_video']=final_video
            with open(user_folder + '/Scores_total_' + str(identifier) + '.txt') as file_in:
                lines = []
                for line in file_in:
                    lines.append(int(line.split(' ')[-1]))
            return render_template('start_loop.html', query_nr=conta, total_queries=n_queries_total, max=np.max(lines), min=np.min(lines), average=np.mean(lines), previous=lines,id_not_found=id_found)

@application.route("/show_video_initial",methods = ['POST','GET'])
@auth.login_required
def show_video_initial():
    identifier=session.get('identifier',None)
    user_folder = here + '/user_' + str(identifier)
    with open(user_folder+'/rng.pkl', 'rb') as inp:
        rng = pickle.load(inp)
    f = open(user_folder + '/save_time.txt', "a")
    f.write('show_test_video_' + str(time.time()) + '\n')
    f.close()
    ###First random query###
    new_idx_train = int(rng.choice(range(n_train_queries), size=n_initial, replace=False))
    new_idx_test = int(rng.choice(range(n_test_queries), size=n_initial, replace=False))
    #save random state

    with open(user_folder + '/rng.pkl', 'wb') as outp:
        pickle.dump(rng, outp, pickle.HIGHEST_PROTOCOL)
    np.save(user_folder + '/idx_train', new_idx_train)  # primo video randomicamente scelto
    np.save(user_folder + '/idx_test', new_idx_test)  # primo video randomicamente scelto
    idx_col_test=np.load(user_folder +'/idx_col_test.npy')
    idx_col_train=np.load(user_folder +'/idx_col_train.npy')
    if train_test_order[conta_first_time]:
        final_video = str(idx_col_train[new_idx_train]) + '.mp4'
    else:
        final_video = str(idx_col_test[new_idx_test]) + '.mp4'
    session['final_video'] = final_video
    return render_template('show_video.html',distorted_video=final_video) ####insert html with video tag here

@application.route("/show_video_loop",methods = ['POST','GET'])
@auth.login_required
def show_video_loop():
    identifier = session.get('identifier', None)
    user_folder = here+'/user_' + str(identifier)
    f = open(user_folder + '/save_time.txt', "a")
    f.write('show_test_video_' + str(time.time()) + '\n')
    f.close()
    conta = int(np.load(user_folder + '/conta.npy'))
    conta_baseline = int(np.load(user_folder + '/conta_baseline.npy'))
    if np.load(user_folder + '/baselines_queries.npy'):
        new_idx_baseline = np.load(user_folder + '/idx_baseline.npy')
        idx_col_train = np.load(user_folder + '/idx_col_train.npy')
        final_video_baseline = str(idx_col_train[new_idx_baseline]) + '.mp4'
        session['final_video'] = final_video_baseline
    elif train_test_order[conta-conta_baseline]: #conta is already at the next experience, I have increased it before
        new_idx_train=np.load(user_folder + '/idx_train.npy')
        idx_col_train = np.load(user_folder + '/idx_col_train.npy')
        final_video_train = str(idx_col_train[new_idx_train]) + '.mp4'
        session['final_video'] = final_video_train
    else:
        new_idx_test=np.load(user_folder + '/idx_test.npy')
        idx_col_test = np.load(user_folder + '/idx_col_test.npy')
        final_video_test = str(idx_col_test[new_idx_test]) + '.mp4'
        session['final_video'] = final_video_test
    return render_template('show_video.html',distorted_video=session.get('final_video',None)) ####insert html with video tag here

@application.route("/show_video_replay",methods = ['POST','GET'])
@auth.login_required
def show_video_replay():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    f = open(user_folder + '/save_time.txt', "a")
    f.write('video_shown_again_' + str(time.time()) + '\n')
    f.close()
    return render_template('show_video_replay.html',distorted_video=session.get('final_video',None)) ####insert html with video tag here

@application.route("/show_video_reference",methods = ['POST','GET'])
@auth.login_required
def show_video_reference():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    f = open(user_folder + '/save_time.txt', "a")
    f.write('reference_video_' + str(time.time()) + '\n')
    f.close()
    video_reference=str(random.choice(reference_list))+'_ref.mp4'
    return render_template('show_video_reference.html',distorted_video=video_reference) ####insert html with video tag here

@application.route("/show_video_reference_score_window",methods = ['POST','GET'])
@auth.login_required
def show_video_reference_score_window():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    f = open(user_folder + '/save_time.txt', "a")
    f.write('reference_video_' + str(time.time()) + '\n')
    f.close()
    video_reference=str(random.choice(reference_list))+'_ref.mp4'
    return render_template('show_video_reference_score_window.html',distorted_video=video_reference) ####insert html with video tag here

@application.route("/show_video_reference_start",methods = ['POST','GET'])
@auth.login_required
def show_video_reference_start():
    identifier = session.get('identifier', None)
    if identifier is None:
        session['time_reference']=str(time.time())
    else:
        user_folder = here + '/user_' + str(identifier)
        f = open(user_folder + '/save_time.txt', "a")
        f.write('reference_video_' + str(time.time()) + '\n')
        f.close()
    video_reference=str(random.choice(reference_list))+'_ref.mp4'
    return render_template('show_video_reference_start.html',distorted_video=video_reference) ####insert html with video tag here

@application.route("/score_window",methods=['POST','GET'])
@auth.login_required
def scores():
    identifier=session.get('identifier',None)
    user_folder = here+'/user_' + str(identifier)
    #calculate and add params
    with open(user_folder + '/Scores_total_' + str(identifier) + '.txt') as file_in:
        lines = []
        for line in file_in:
            lines.append(int(line.split(' ')[-1]))

    if lines!=[]:
        return render_template('Scores_interface.html', max=np.max(lines), min=np.min(lines), average=np.mean(lines), previous=lines)
    else:
        return render_template('Scores_interface.html', max=None, min=None,average=None, previous=None)

@application.route("/paused_experience",methods=['POST','GET'])
@auth.login_required
def pause_exp():
    identifier = session.get('identifier', None)
    user_folder = here+'/user_' + str(identifier)
    f = open(user_folder+'/save_time.txt', "a")
    f.write('pause_experience'+str(time.time()) +'\n')
    f.close()
    return render_template('paused_experience.html',identifier=identifier)

@application.route("/save_controls",methods=['POST'])
@auth.login_required
def save_contr():
    keyword = request.form.get('keyword')
    identifier=session.get('identifier',None)
    if identifier is None:
        return ('', 204)
    user_folder = here + '/user_' + str(identifier)
    f = open(user_folder + '/save_time.txt', "a")
    where=keyword.split('_')[0]
    if where in ['replay','reference']:
        where=where+'_'
    else:
        where=''
    print(where)
    if keyword==where+'play':
        f.write(where+'play_' + str(time.time()) + '\n')
    elif keyword==where+'paused':
        f.write(where+'pause_video_' + str(time.time()) + '\n')
    elif keyword==where+'seeked':
        f.close()
        with open(user_folder + '/save_time.txt', 'r+') as f:
        # read an store all lines into list
            lines = f.readlines()
            # move file pointer to the beginning of a file
            f.seek(0)
            # truncate the file
            f.truncate()

            # start writing lines except the last line
            # lines[:-1] from line 0 to the second last line
            f.writelines(lines[:-1])
        f = open(user_folder + '/save_time.txt', "a")
        f.write(where+'seeked_' + str(time.time()) + '\n')
    elif keyword=='statistics':
        f.write('statistics_' + str(time.time()) + '\n')
    elif keyword=='ended':
        f.close()
        with open(user_folder + '/save_time.txt', 'r+') as f:
            # read an store all lines into list
            lines = f.readlines()
            # move file pointer to the beginning of a file
            f.seek(0)
            # truncate the file
            f.truncate()

            # start writing lines except the last line
            # lines[:-1] from line 0 to the second last line
            f.writelines(lines[:-1])

    f.close()

    return ('', 204)

@application.route("/redirect_reference_start_loop",methods=['POST','GET'])
@auth.login_required
def redirect_after_reference_start_loop():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    conta = int(np.load(user_folder + '/conta.npy'))
    # calcola le statistiche delle precedenti queries
    with open(user_folder + '/Scores_total_' + str(identifier) + '.txt') as file_in:
        lines = []
        for line in file_in:
            lines.append(int(line.split(' ')[-1]))

    return render_template('start_loop.html', query_nr=conta, total_queries=n_queries_total, max=np.max(lines),
                           min=np.min(lines), average=np.mean(lines), previous=lines)

@application.route("/capture_info_form",methods=['POST','GET'])
@auth.login_required
def capture_form():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    # capture values from form
    homecountry = request.form.get('country')
    asscountry = request.form.get('asscountry')
    gender = request.form.get('gender')
    age = request.form.get('age')
    device = request.form.get('Device')
    suggestions = request.form.get('suggestions')
    annoying = request.form.get('Annoying')

    f = open(user_folder + '/save_personal_info.txt', "a")

    #save values in file
    f.write('homecountry_' + str(homecountry) + '\n')
    f.write('asscountry_' + str(asscountry) + '\n')
    f.write('gender_' + str(gender) + '\n')
    f.write('age_' + str(age) + '\n')
    f.write('device_' + str(device) + '\n')
    f.write('annoying_' + str(annoying) + '\n')
    f.write('suggestions_' + str(suggestions) + '\n')
    f.close()
    return render_template('end.html')

@application.route("/capture_info_browser", methods=['POST', 'GET'])
@auth.login_required
def capture_browser():
    identifier = session.get('identifier', None)
    user_folder = here + '/user_' + str(identifier)
    data = request.get_json()
    bro=data['bro']
    wid=data['wid']
    heig=data['heig']
    f = open(user_folder + '/save_personal_info.txt', "a")
    f.write('browser_' + str(bro) + '\n')
    f.write('width_' + str(wid) + '\n')
    f.write('height_' + str(heig) + '\n')
    f.close()
    return ('', 204)

if __name__== '__main__':
    application.run(host='0.0.0.0', port=7000, debug=True, threaded=True)