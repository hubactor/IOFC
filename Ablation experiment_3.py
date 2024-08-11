import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

max_iter_1 = 100
fit_max = 1
fit_min = -1
# Probability of generating interconnection and perturbation
Interconnection_prob = random.random()
Perturbation_prob = random.random()
def KNN(data, k):
    knn_model = NearestNeighbors(n_neighbors=2 * k + 1)
    knn_model.fit(data)
    distances, indices = knn_model.kneighbors(data)
    return distances, indices


def RNN(data, indices):
    rnn = {}
    for i in range(len(data)):
        rnn[i] = set()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            rnn[neighbor].add(i)
    return rnn

def select_centroid(data, k):
    distance, indice = KNN(data, k)
    receive_rnn = RNN(data, indice)
    centers = []
    for index, neighbors in receive_rnn.items():
        local_density = len(neighbors)
        centers.append(local_density)
    select_center = np.argsort(centers)[-k:]
    return data[select_center]

def Interconnection(per_centroid1, per_centroid2, data):
    p_rand = np.random.rand()
    if p_rand < Interconnection_prob:
        inter_point = np.random.randint(1, data.shape[1])
        Join1 = np.concatenate((per_centroid1[:inter_point], per_centroid2[inter_point:]))
        Join2 = np.concatenate((per_centroid2[:inter_point], per_centroid1[inter_point:]))
    else:
        Join1, Join2 = per_centroid1.copy(), per_centroid2.copy()
    return Join1, Join2

def init_cluster(data, k):
    centroid = select_centroid(data, k)
    for iter in range(max_iter_1):
        compactness_all = np.zeros(k)
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
            compactness_all[i] = compactness
        max_compactness = np.argsort(compactness_all)[-(k - 1):][::-1]
        min_compaceness = np.argsort(compactness)[0]
        max_ph_centroid = centroid[max_compactness]
        min_ph_centroid = centroid[min_compaceness]
        # print(residue)
        length = len(max_ph_centroid)
        new_centroid = []
        while len(new_centroid) < length:
            old_centroid_index = np.random.choice(len(max_ph_centroid), size=2, replace=False)
            c1, c2 = Interconnection(max_ph_centroid[old_centroid_index[0]], max_ph_centroid[old_centroid_index[1]], data)
            new_centroid.extend([c1, c2])
        new_centroid.extend(min_ph_centroid)
        new_centroid = np.array(new_centroid)
        # Ensure the number of centroid points
        if len(new_centroid) > k:
            new_centroid = np.delete(new_centroid, -2, axis=0)
        if len(new_centroid) < k:
            continue
        for i in range(k):
            centroid[i] = new_centroid[i]
        if np.sum((pre_centroid - centroid) ** 2) <= 0.001:
            break
    return init_label




