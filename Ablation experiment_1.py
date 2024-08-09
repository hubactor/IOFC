import numpy as np

max_iter_1 = 100
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
    for index , neighbors in  receive_rnn.items():
        local_density = len(neighbors)
        centers.append(local_density)
    select_center = np.argsort(centers)[-k:]
    # print(select_center)
    return data[select_center]

def init_cluster(data, k):
    centroid= select_centroid(data, k)
    # print(centroid)
    for iter in range(max_iter_1):
        init_label = np.zeros(data.shape[0])
        pre_centroid = np.array(centroid)
        for locate, point in enumerate(data):
            distance = []
            for cent in centroid:
                distance.append(np.linalg.norm(point - cent))
            index = np.argmin(distance)
            init_label[locate] = index
        for i in range(k):
            cluster = data[init_label == i]
            num = cluster.shape[0]
            for p in range(cluster.shape[1]):
                sum = np.sum(cluster[:, p])
                new_center = sum / num
                centroid[i, p] = new_center
        if np.sum(np.abs(pre_centroid - centroid)) <= 0.001:
            break
    return init_label