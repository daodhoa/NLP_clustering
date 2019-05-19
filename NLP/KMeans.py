import numpy as np
from sklearn.metrics import pairwise_distances

class Kmeans:
    def __init__(self, k, seed=None, max_iter=200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialise_centroids(self, data):
        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids

    def assign_clusters(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(data, self.centroids, metric=jaccard_metric)
        self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    def predict(self, data):
        return self.assign_clusters(data)


    def fit_kmeans(self, data):
        self.centroids = self.initialise_centroids(data)
        centers = [self.centroids]
        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            if has_converged(centers[-1], self.centroids):
                print("converged at iter: " + str(iter))
                break
            centers.append(self.centroids)
            if iter % 100 == 0:
                print("Running Model Iteration %d " % iter)
        print("Model finished running")
        return self

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))

import math
def cosine_metric(vector1, vector2):
    tuso = 0
    bp_dodai_v1 = 0
    bp_dodai_v2 = 0
    for i in range(vector1.shape[0]):
        tuso += (vector1[i] * vector2[i])
        bp_dodai_v1 +=  vector1[i] ** 2
        bp_dodai_v2 +=  vector2[i] ** 2
    mauso  = math.sqrt(bp_dodai_v1) * math.sqrt(bp_dodai_v2)
    if mauso ==0:
        return 0
    return tuso/mauso

def jaccard_metric(vector1, vector2):
    tuso = 0
    bp_dodai_v1 = 0
    bp_dodai_v2 = 0
    for i in range(vector1.shape[0]):
        tuso += (vector1[i] * vector2[i])
        bp_dodai_v1 += vector1[i] ** 2
        bp_dodai_v2 += vector2[i] ** 2
    mauso = bp_dodai_v1 + bp_dodai_v2 - tuso
    if mauso ==0: return 0
    return (1 - tuso / mauso)

def euclid_metric(vector1, vector2):
    bp_dodai = 0
    for i in range(vector1.shape[0]):
        bp_dodai += (vector1[i] - vector2[i]) **2
    return bp_dodai**(1/2)
