import sys
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import time
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform, pdist
from numpy import array_equal
from sklearn.metrics import silhouette_score
max_iter_1 = 150
fit_max = 1
fit_min = -1
interconnection_prob = 0.6
perturbation_prob = 0.8
def KNN(data, k):
    knn_model = NearestNeighbors(n_neighbors=2 * k + 1)
    knn_model.fit(data)
    distances, indices = knn_model.kneighbors(data)
    return distances,indices
def RNN(data, indices):
    rnn = {}
    for i in range(len(data)):
        rnn[i] = set()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            rnn[neighbor].add(i)
    return rnn
def select_centroid(data,k):
    distance,indice = KNN(data,k)
    receive_rnn = RNN(data,indice)
    centers = []
    for index,neighbors in  receive_rnn.items():
        local_density = len(neighbors)
        centers.append(local_density)
    select_center = np.argsort(centers)[-k:]
    return data[select_center]
def Interconnection(old_center1, old_center2,data):
    p_rand = np.random.rand()
    if p_rand < interconnection_prob:
        interconnection = np.random.randint(1, data.shape[1])
        c1 = np.concatenate((old_center1[:interconnection], old_center2[interconnection:]))
        c2 = np.concatenate((old_center1[:interconnection], old_center2[interconnection:]))
    else:
        c1, c2 = old_center1.copy(), old_center2.copy()
    return c1, c2
def Perturbation(c,data):
    for i in range(data.shape[1]):
        q = np.random.rand()
        if q < perturbation_prob:
            c[i] += np.random.normal(0,0.1)
    return c
def init_cluster(data, k):
    centroid= select_centroid(data, k)
    for iter in range(max_iter_1):
        distance_mean = np.zeros(k)
        init_label = np.zeros(data.shape[0])
        pre_centroid = np.array(centroid)
        for locate, point in enumerate(data):
            distance = []
            for cent in centroid:
                distance.append(np.linalg.norm(point - cent))
            index = np.argmin(distance)
            init_label[locate] = index
        for i in range(k):
            cluster_indice = np.where(init_label == i)[0]
            if len(cluster_indice) == 0:
                break
            cluster = data[cluster_indice]
            distance_mean_value = np.linalg.norm(centroid[i] - cluster) / len(cluster)
            compactness = (fit_max - distance_mean_value) / (fit_max - fit_min)
            distance_mean[i] = compactness
        max_locate = np.argsort(distance_mean)[-(k - 1):][::-1]
        max_cp_center = centroid[max_locate]
        residue = []
        for p in centroid:
            found =False
            for q in max_cp_center:
                if np.array_equal(p,q):
                    found =True
                    break
            if not found:
                residue.append(p)
        length = len( max_cp_center)
        new_center = []
        while len(new_center) < length :
             old_center = np.random.choice(len( max_cp_center),size=2,replace=False)
             c_1,c_2 = Interconnection( max_cp_center[old_center[0]], max_cp_center[old_center[1]],data)
             c_1 = Perturbation(c_1,data)
             c_2 = Perturbation(c_2,data)
             new_center.extend([c_1,c_2])
        new_center.extend(residue)
        new_center = np.array(new_center)
        if len(new_center) < k:
            continue
        for i in range(k):
            centroid[i] = new_center[i]
        if np.sum(np.abs(pre_centroid - centroid)) == 0:
            break
    return init_label



