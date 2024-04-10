import math
import sys
import numpy as np
import random
from sklearn.preprocessing import scale
from scipy.stats import anderson
from sklearn.cluster import MiniBatchKMeans, KMeans

class acc_means(object):

    def __init__(self, data, n_clusters):
        super(acc_means, self).__init__()
        self.data_list = data
        self.n_clusters = n_clusters
        self.DIMENSION = data.shape[1]
        self.N = data.shape[0]


    def getLoss2(self, data, c, center, n_cluster):
        val = 0
        for i in range(self.N):
            for k in range(n_cluster):
                if c[i] == k:
                    val = val + np.linalg.norm(data[i] - center[k]) ** 2
                    break
        return val

    def get_center_distance(self, centroid):
        l = len(centroid)
        result = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                if j >= i:
                    result[i][j] = np.linalg.norm(centroid[i] - centroid[j])
                else:
                    result[i][j] = result[j][i]
        return result

    def get_s(self, c, c_dist, n_cluster):
        res = c_dist[c][0]
        if c == 0:
            res = c_dist[c][1]

        for i in range(n_cluster):
            if c_dist[c][i] < res and c != i:
                res = c_dist[c][i]
        return res / 2

    def accelerated_kmeans(self, data, n_cluster, centroid, tol=1e-6):
        l = np.zeros([self.N, n_cluster])  # Lower bound matrix
        u = np.zeros(self.N)  # Upper bound vector
        c = np.zeros(self.N)  # assignments of data points

        d_ptc = list()

        for i in range(self.N):
            dist = []
            # vector_i = data[i]
            # min_index = 0
            for j in range(n_cluster):
                d_xc = np.linalg.norm(data[i] - centroid[j])
                dist.append(d_xc)
                l[i][j] = d_xc
            d_ptc.append(dist)

            min_dist = min(dist)
            min_index = dist.index(min_dist)
            c[i] = min_index
            u[i] = min_dist

        r = [False for i in range(self.N)]
        loss_prev = self.getLoss2(data, c, centroid, n_cluster)
        ite = 0
        while (True):
            ite += 1
            ## step 1
            d_center = self.get_center_distance(centroid)  # Distances between centers
            s = [self.get_s(i, d_center, n_cluster) for i in range(n_cluster)]

            x = []
            for i in range(self.N):
                if u[i] <= s[int(c[i])]:  ## step 2
                    x.append(i)
                else:
                    for j in range(n_cluster):  ## step 3
                        ## 3 (i) (ii) (iii)
                        if j != c[i] and u[i] > l[i][j] and u[i] > 0.5 * d_center[int(c[i])][j]:
                            ## 3a
                            if r[i]:
                                d_xc = np.linalg.norm(data[i] - centroid[int(c[i])])
                                d_ptc[i][int(c[i])] = d_xc  ####
                                r[i] = False
                            else:
                                d_ptc[i][int(c[i])] = u[i]
                            ## 3b
                            if d_ptc[i][int(c[i])] > l[i][j] or \
                                    d_ptc[i][int(c[i])] > 0.5 * d_center[int(c[i]), j]:
                                d_xc = np.linalg.norm(data[i] - centroid[j])
                                if d_xc < d_ptc[i][int(c[i])]:
                                    # d_ptc[i][j] = d_xc ####
                                    c[i] = j
            ## step 4
            flag = 0
            m = []
            for k in range(n_cluster):
                cnt = 0
                vec = np.zeros(self.DIMENSION)
                for i in range(self.N):
                    if c[i] == k:
                        cnt = cnt + 1
                        vec = np.add(vec, data[i])
                # print("k = {} and cnt = {}".format(k,cnt))
                if (cnt == 0):  ## Bad centroid -> Initialize again
                    centroid = self.initialize(self.data_list, n_cluster)
                    flag = 1
                    break
                m.append(vec / cnt)
            if flag == 1:
                l = np.zeros([self.N, n_cluster])  # Lower bound matrix
                u = np.zeros(self.N)  # Upper bound vector
                c = np.zeros(self.N)  # assignments of data points

                d_ptc = list()

                for i in range(self.N):
                    dist = []

                    for j in range(n_cluster):
                        d_xc = np.linalg.norm(data[i] - centroid[j])
                        dist.append(d_xc)
                        l[i][j] = d_xc
                    d_ptc.append(dist)

                    min_dist = min(dist)
                    min_index = dist.index(min_dist)
                    c[i] = min_index
                    u[i] = min_dist

                r = [False for i in range(self.N)]
                loss_prev = self.getLoss2(data, c, centroid, n_cluster)
                flag = 0
                continue

            ## step 5
            for i in range(self.N):
                for j in range(n_cluster):
                    d_cmc = np.linalg.norm(centroid[j] - m[j])
                    diff = l[i][j] - d_cmc
                    if diff <= 0:
                        l[i][j] = 0
                    else:
                        l[i][j] = diff

            ## step 6
            for i in range(self.N):
                u[i] = u[i] + np.linalg.norm(centroid[int(c[i])] - m[int(c[i])])
                r[i] = True

            ## step 7
            centroid = m
            loss_next = self.getLoss2(data, c, centroid, n_cluster)

            ## Check convergence
            if (abs(loss_prev - loss_next) <= tol):
                return c, centroid, ite
            loss_prev = loss_next

    def distance(self, p1, p2):
        return np.sum((p1 - p2) ** 2)

    def initialize(self, data, k):
        ## initialize the centroids list and add
        ## a randomly selected data point to the list
        centroids = []
        centroids.append(data[random.randint(0, len(data) - 1)])
        # plot(data, np.array(centroids))

        ## compute remaining k - 1 centroids
        for c_id in range(k - 1):

            ## initialize a list to store distances of data
            ## points from nearest centroid
            dist = []
            for i in range(len(data)):
                point = data[i]
                d = sys.maxsize

                ## compute distance of 'point' from each of the previously
                ## selected centroid and store the minimum distance
                for j in range(len(centroids)):
                    temp_dist = self.distance(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)

            ## select data point with maximum distance as our next centroid
            dist = np.array(dist)
            next_centroid = data[np.argmax(dist)]
            centroids.append(next_centroid)
            dist = []
            # plot(data, np.array(centroids))
        return centroids

    def getdata_index(self, point, cent):
        dist = []
        for i in range(len(cent)):
            dist.append(math.sqrt(np.sum((point - cent[i]) ** 2)))

        return dist.index(min(dist))



    def fit(self):

        centroid = self.initialize(self.data_list, self.n_clusters)
        cent = centroid[:]
        _, center, _ = self.accelerated_kmeans(self.data_list, self.n_clusters, cent)
        self.center = center

        data_index = np.array([(i, False) for i in range(self.data_list.shape[0])])
        for i in range(self.data_list.shape[0]):
            data_index[i][1] = self.getdata_index(self.data_list[i], center)
        self.labels_ = data_index[:, 1]


class GMeans(object):
    """strictness = how strict should the anderson-darling test for normality be
            0: not at all strict
            4: very strict
    """

    def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):

        super(GMeans, self).__init__()

        self.max_depth = max_depth

        self.min_obs = min_obs

        self.random_state = random_state

        if strictness not in range(5):
            raise ValueError("strictness parameter must be integer from 0 to 4")
        self.strictness = strictness

        self.stopping_criteria = []

    def _gaussianCheck(self, vector):
        """
        check whether a given input vector follows a gaussian distribution
        H0: vector is distributed gaussian
        H1: vector is not distributed gaussian
        """
        output = anderson(vector)

        if output[0] <= output[1][self.strictness]:
            return True
        else:
            return False

    def _recursiveClustering(self, data, depth, index):
        """
        recursively run kmeans with k=2 on your data until a max_depth is reached or we have
            gaussian clusters
        """
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('max_depth')
            return

        km = MiniBatchKMeans(n_clusters=2, random_state=self.random_state)
        km.fit(data)

        centers = km.cluster_centers_
        v = centers[0] - centers[1]
        x_prime = scale(data.dot(v) / (v.dot(v)))
        gaussian = self._gaussianCheck(x_prime)

        # print gaussian

        if gaussian == True:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('gaussian')
            return

        labels = set(km.labels_)
        for k in labels:
            current_data = data[km.labels_ == k]

            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append('min_obs')
                return

            current_index = index[km.labels_ == k]
            current_index[:, 1] = np.random.randint(0, 100000000)
            self._recursiveClustering(data=current_data, depth=depth, index=current_index)

    # set_trace()

    def fit(self, data):
        """
        fit the recursive clustering model to the data
        """
        self.data = data

        data_index = np.array([(i, False) for i in range(data.shape[0])])
        self.data_index = data_index

        self._recursiveClustering(data=data, depth=0, index=data_index)

        self.labels_ = self.data_index[:, 1]
        diuniq = np.unique(self.data_index[:, 1])
        arr = dict(zip(diuniq, range(0, len(diuniq))))
        for i in range(0, len(self.labels_)):
            self.labels_[i] = arr[self.labels_[i]]


class GMeans_p(object):
    """strictness = how strict should the anderson-darling test for normality be
            0: not at all strict
            4: very strict
    """

    def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):

        super(GMeans_p, self).__init__()

        self.max_depth = max_depth

        self.min_obs = min_obs

        self.random_state = random_state

        if strictness not in range(5):
            raise ValueError("strictness parameter must be integer from 0 to 4")
        self.strictness = strictness

        self.stopping_criteria = []

    def _gaussianCheck(self, vector):
        """
        check whether a given input vector follows a gaussian distribution
        H0: vector is distributed gaussian
        H1: vector is not distributed gaussian
        """
        output = anderson(vector)

        if output[0] <= output[1][self.strictness]:
            return True
        else:
            return False

    def _recursiveClustering(self, data, depth, index):
        """
        recursively run kmeans with k=2 on your data until a max_depth is reached or we have
            gaussian clusters
        """
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('max_depth')
            return

        km = acc_means(data, 2)
        km.fit()

        centers = km.center
        v = centers[0] - centers[1]
        x_prime = scale(data.dot(v) / (v.dot(v)))
        gaussian = self._gaussianCheck(x_prime)

        # print gaussian

        if gaussian == True:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('gaussian')
            return

        labels = set(km.labels_)
        for k in labels:
            current_data = data[km.labels_ == k]

            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append('min_obs')
                return

            current_index = index[km.labels_ == k]
            current_index[:, 1] = np.random.randint(0, 2 ** 31 - 1)
            self._recursiveClustering(data=current_data, depth=depth, index=current_index)

    # set_trace()

    def fit(self, data):
        """
        fit the recursive clustering model to the data
        """
        self.data = data

        data_index = np.array([(i, False) for i in range(data.shape[0])])
        self.data_index = data_index

        self._recursiveClustering(data=data, depth=0, index=data_index)

        self.labels_ = self.data_index[:, 1]
        diuniq = np.unique(self.data_index[:, 1])
        arr = dict(zip(diuniq, range(0, len(diuniq))))
        for i in range(0, len(self.labels_)):
            self.labels_[i] = arr[self.labels_[i]]


class GMeans_pp(object):
    """strictness = how strict should the anderson-darling test for normality be
            0: not at all strict
            4: very strict
    """

    def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):

        super(GMeans_pp, self).__init__()

        self.max_depth = max_depth

        self.min_obs = min_obs

        self.random_state = random_state

        if strictness not in range(5):
            raise ValueError("strictness parameter must be integer from 0 to 4")
        self.strictness = strictness

        self.stopping_criteria = []

    def _gaussianCheck(self, vector):
        """
        check whether a given input vector follows a gaussian distribution
        H0: vector is distributed gaussian
        H1: vector is not distributed gaussian
        """
        output = anderson(vector)

        if output[0] <= output[1][self.strictness]:
            return True
        else:
            return False

    def _recursiveClustering(self, data, depth, index):
        """
        recursively run kmeans with k=2 on your data until a max_depth is reached or we have
            gaussian clusters
        """
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('max_depth')
            return

        km = KMeans(n_clusters=2, random_state=self.random_state)
        km.fit(data)

        centers = km.cluster_centers_
        v = centers[0] - centers[1]
        x_prime = scale(data.dot(v) / (v.dot(v)))
        gaussian = self._gaussianCheck(x_prime)

        # print gaussian

        if gaussian == True:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('gaussian')
            return

        labels = set(km.labels_)
        for k in labels:
            current_data = data[km.labels_ == k]

            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append('min_obs')
                return

            current_index = index[km.labels_ == k]
            current_index[:, 1] = np.random.randint(0, 100000000)
            self._recursiveClustering(data=current_data, depth=depth, index=current_index)

    # set_trace()

    def fit(self, data):
        """
        fit the recursive clustering model to the data
        """
        self.data = data

        data_index = np.array([(i, False) for i in range(data.shape[0])])
        self.data_index = data_index

        self._recursiveClustering(data=data, depth=0, index=data_index)

        self.labels_ = self.data_index[:, 1]
        diuniq = np.unique(self.data_index[:, 1])
        arr = dict(zip(diuniq, range(0, len(diuniq))))
        for i in range(0, len(self.labels_)):
            self.labels_[i] = arr[self.labels_[i]]


