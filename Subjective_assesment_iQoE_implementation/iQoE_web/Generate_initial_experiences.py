import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy
from sklearn.metrics import silhouette_score

#insert path to your folder Figure iQoE_web

#const
nr_c=4
seed=2
n_queries_train=50
n_queries_test=30
synthetic_experiences_ = np.load('iQoE_synth_exp.npy')
synthetic_experiences = np.delete(synthetic_experiences_, -1, axis=1)
# add a column to monitor the original index of the experiences
idx_col=[i for i in range(len(synthetic_experiences_))]

#select the main features on which to do the clustering
all_features=copy.copy(synthetic_experiences)
two_features=[]#sum bitrate sum rebuff bitrate diff total
for exp in synthetic_experiences:
    rep = []
    for i in range(0, (nr_c * 10 - 1), 10):
        rep.append(float(exp[i]))
    s_rep = np.array(rep).sum()

    bit = []
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
    # sumbit
    s_bit = np.array(bit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_rep = np.abs(np.array(rep[1:]) - np.array(rep[:-1])).sum()

    # collection
    two_features.append([s_rep, s_reb, s_dif_rep])
two_features=np.array(two_features).reshape(-1,3)

#elbow
distortions = []
silhouette_avg = []
K = range(2,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(two_features)
    distortions.append(kmeanModel.inertia_)
    cluster_labels = kmeanModel.labels_

    silhouette_avg.append(silhouette_score(two_features, cluster_labels))

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#######################################################
#kmeans with best k
k=5
kmeanModel = KMeans(n_clusters=k)
kmeanModel.fit(two_features)
cluster_labels = kmeanModel.labels_ #those basically are my "ground truth"
#####################################################
#attach idx_col to synth_exp_ and use cluster_labels as y and run a stratified version of train test split
synth_exp_with_idx=np.c_[ synthetic_experiences_, idx_col ]#cluster_labels
se_train, se_test, cluster_labels_train, cluster_labels_test= train_test_split(synth_exp_with_idx, cluster_labels,
                                   random_state=seed,
                                   test_size=0.03,
                                   shuffle=True,
                                   stratify=cluster_labels)
#remove the idx_col
idx_train_fl=se_train[:,-1]
idx_train=[int(i) for i in idx_train_fl]
se_train = np.delete(se_train, -1, axis=1)
idx_test_fl=se_test[:,-1]
idx_test=[int(i) for i in idx_test_fl]
se_test = np.delete(se_test, -1, axis=1)
#####################################################################################
if os.path.exists("test_description.txt"):
    os.remove("test_description.txt")
if os.path.exists("train_description.txt"):
    os.remove("train_description.txt")
#watch the train/test data
f = open("train_description.txt", "a+")
for idx_exp,each_exp in enumerate(se_train): #todo modifica rep che ora va da 0 a 12
    f.write(str(idx_train[idx_exp])+' ')
    for i in range(0,nr_c*10,10):
        rep=each_exp[i]
        sta=each_exp[i+1]
        #print(rep)
        #write in file
        f.write(str(rep) + '-' + str(round(sta,2)) +'--->')
    f.write('\n')
f.close()

print('stall sum for test')
#rep--stall-----rep--stall-------
f = open("test_description.txt", "a+")
for idx_exp,each_exp in enumerate(se_test): #todo modifica rep che ora va da 0 a 12
    sta_sum = []
    f.write(str(idx_test[idx_exp])+' ')
    for i in range(0,nr_c*10,10):
        rep=each_exp[i]
        sta=each_exp[i+1]
        sta_sum.append(sta)
        #print(rep)
        #write in file
        f.write(str(rep) + '-' + str(round(sta,2)) +'--->')
    f.write('\n')
    print(np.sum(sta_sum))
f.close()

######################################################################################
if not os.path.exists('original_database'):
    os.makedirs('original_database')
np.save('original_database/synth_exp_train',se_train)
np.save('original_database/idx_col_train',idx_train)
np.save('original_database/synth_exp_test',se_test)
np.save('original_database/idx_col_test',idx_test)

array1=[True for i in range(n_queries_train)]
array2=[False for i in range(n_queries_test)]
array=array1+array2
random.Random(seed).shuffle(array)
np.save('original_database/train_test_order',array)

## generate scaled data
# remove last column that indicate the nr of initial chunk and the idx of the experience
X_train = np.delete(se_train, -1, axis=1)
X_test = np.delete(se_test, -1, axis=1)
# scale video pool features
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
np.save('original_database/X_train_scaled', X_train_scaled)
np.save('original_database/X_test_scaled', X_test_scaled)

# matplotlib histogram
# plt.hist(flights['arr_delay'], color = 'blue', edgecolor = 'black',
#          bins = int(180/5))