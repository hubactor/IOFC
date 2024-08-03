import math
import os
import math
import random
import sys
from collections import Counter
import numpy as np
import umap
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KernelDensity
import pandas as pd
import seaborn as sns
import warnings
import time
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform, pdist
from numpy import array_equal
from sklearn.metrics import silhouette_score
from umap import UMAP
warnings.filterwarnings("ignore")
pathData = r"D:\improve k-means\dataset\D1.txt"
data = loadtxt(pathData)
label = data[:, -1]
dataset = data[:, :-1]
# 交叉概率
crossover_prob = 0.6
# 设置变异概率
mutation_prob = 0.8
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
# 进行后代的产生
def crossover(parent1, parent2,data):
    p_rand = np.random.rand()
    if p_rand < crossover_prob:
        crossover_point = np.random.randint(1, data.shape[1])
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2

# 对新产生的子代进行随机变异
def mutation(children,data):
    # 对数据点的每一维度添加随机扰动
    for i in range(data.shape[1]):
        q = np.random.rand()
        if q < mutation_prob:
            children[i] += np.random.normal(0,0.1)
    return children

def init_cluster(data, k):
    max_iter_1 = 150
    # 将聚类中心设置为黑寡妇蛛
    fit_max = 1
    fit_min = -1
    centroid= select_centroid(data, k)
    # print(centroid)
    for iter in range(max_iter_1):
        distance_mean = np.zeros(k)
        # m = -0.1
        init_label = np.zeros(data.shape[0])
        pre_centroid = np.array(centroid)
        # 对数据进行聚类
        for locate, point in enumerate(data):
            distance = []
            for cent in centroid:
                distance.append(np.linalg.norm(point - cent))
            index = np.argmin(distance)
            init_label[locate] = index
        # 以每个中心到聚类的平均距离为适应度值
        for i in range(k):
            cluster_indice = np.where(init_label == i)[0]
            if len(cluster_indice) == 0:
                break
            cluster = data[cluster_indice]
            distance_mean_value = np.linalg.norm(centroid[i] - cluster) / len(cluster)
            pheromone = (fit_max - distance_mean_value) / (fit_max - fit_min)
            distance_mean[i] = pheromone
        # 对适应度值进行排序
        # 获取至少两个及以上的最高适应度中心:
        max_pheromone = np.argsort(distance_mean)[-(k - 1):][::-1]
        # 获取中心
        max_ph_center = centroid[max_pheromone]
        # print(max_ph_center)
        residue = []
        # 将剩余1个中心放入到residue中
        for p in centroid:
            found =False
            for q in max_ph_center:
                if np.array_equal(p,q):
                    found =True
                    break
            if not found:
                residue.append(p)
        length = len(max_ph_center)
        # 用于存放新的中心
        new_center = []
        while len(new_center) < length :
             #从之前选去的适应度中心的数组中，随机选取两个作为父母
             parents = np.random.choice(len(max_ph_center),size=2,replace=False)
             #产生两个子代
             child1,child2 = crossover(max_ph_center[parents[0]],max_ph_center[parents[1]],dataset)
             # 子代进行随机变异
             child1 = mutation(child1,data)
             child2 = mutation(child2,data)
             #将变异后的中心放入列表中
             new_center.extend([child1,child2])
        # 将新中心转换成数组
        new_center.extend(residue)
        new_center = np.array(new_center)
        if len(new_center) < k:
            continue
        for i in range(k):
            centroid[i] = new_center[i]
        if np.sum(np.abs(pre_centroid - centroid)) == 0:
            break
    return init_label


# for k in range(3,7):
k = 3
# start = time.time()
# label_2 = init_cluster(dataset, k)
# end = time.time()
# final = end - start
# print(final)
# print("k=",k)
crossover_prob = 0.1
while crossover_prob <1:
    print("crossover_prob=",crossover_prob)
    mutation_prob = 0.1
    while mutation_prob < 1:
        print("mutation_prob=",mutation_prob)
        x = []
        y = []
        for i in range(100):
            label_2 = init_cluster(dataset, k)
            ari = adjusted_rand_score(label,label_2)
            nmi = normalized_mutual_info_score(label, label_2)
                # print("ari =", ari)
            y.append(nmi)
            x.append(ari)
        mutation_prob += 0.1
        print("max.ari =", max(x))
        print("max.nmi =",max(y))
    crossover_prob += 0.1
#     ari = adjusted_rand_score(label, label_2)
#     print(ari)
'''
ari = []
nmi = []
for i in range(2,k):
    label_2 = init_cluster(dataset, i)
    ari_2 = adjusted_rand_score(label_2, label)
    nmi_2 = normalized_mutual_info_score(label_2,label)
    ari.append(ari_2)
    nmi.append(nmi_2)
    print("i= ",i)
    print("ari =", ari_2)
    print("nmi=",nmi_2)
#
# print(max(ari))
# print(max(nmi))
'''

