# -*- coding: utf-8 -*-
# @time       : 5/11/2019 10:09 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : KMeans.py
# @description: 


from random import seed, randint, random
from collections import Counter
import time


# 运行时间
def run_time(fn):
    def fun():
        start = time()
        fn()
        ret = time() - start
        ret *= 1e3
        print("Total run time is %.1f ms\n" % ret)
    return fun()


def load_data(data_path):
    f = open(data_path)
    X = []
    y = []
    for line in f:
        line = line[:-1].split(',')
        xi = [float(s) for s in line[:-1]]
        yi = line[-1]
        if '.' in yi:
            yi = float(yi)
        else:
            yi = int(yi)
        X.append(xi)
        y.append(yi)
    f.close()
    return X, y


def get_euclidean_distance(arr1, arr2):
    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5


def get_cosine_distance(arr1, arr2):
    numerator = sum(x1 * x2 for x1, x2 in zip(arr1, arr2))
    denominator = (sum(x1 ** 2 for x1 in arr1) * sum(x2 ** 2 for x2 in arr2)) ** 0.5
    return numerator / denominator


class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=42, distance_fn="eu"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        assert distance_fn in ("eu", "cos"), "distance_fn must be eu or cos"
        if distance_fn == "eu":
            self.distance_fn = get_euclidean_distance
        if distance_fn == "cos":
            self.distance_fn = get_cosine_distance
        self.cluster_centers = None
        self.cluster_samples_cnt = None
        self.empty_cluster_idxs = None
        self.n_features = 0
        seed(random_state)

    def bin_search(self, X, lst):
        low, high = 0, len(lst) - 1
        assert lst[low] <= X <= lst[high], "target not in lst"
        while True:
            mid = (low + high) // 2
            if mid == 0 or X >= lst[mid]:
                low = mid + 1
            elif X < lst[mid - 1]:
                high = mid - 1
            else:
                break
        return mid

    def cmp_arr(self, arr1, arr2, eps=1e-8):
        return len(arr1) == len(arr2) and all(abs(a - b) < eps for a, b in zip(arr1, arr2))

    def init_cluster_centers(self, X):
        n = len(X)
        centers = [X[randint(0, n-1)]]
        for _ in range(self.n_clusters-1):
            center_pre = centers[-1]
            idx_dists = ([i, self.distance_fn(Xi, center_pre)] for i, Xi in enumerate(X))
            idx_dists = sorted(idx_dists, key=lambda x: x[1])
            dists = [idx_dist[1] for idx_dist in idx_dists]
            total = sum(dists)
            for i in range(1, n):
                dists[i] /= total
                dists[i] += dists[i-1]
            while True:
                num = random()
                dist_idx = self.bin_search(num, dists)
                row_idx = idx_dists[dist_idx][0]
                cur_center = X[row_idx]
                if not any(self.cmp_arr(cur_center, center) for center in centers):
                    break
            centers.append(cur_center)
        return centers

    def get_nearest_center(self, Xi):
        return min(((i, self.distance_fn(Xi, center)) for i, center in enumerate(self.cluster_centers)), key=lambda x: x[1])[0]

    def get_nearest_centers(self, X):
        return [self.get_nearest_center(Xi) for Xi in X]

    def get_empty_cluster_idxs(self):
        clusters = [(i, self.cluster_samples_cnt[i]) for i in range(self.n_clusters)]
        empty_clusters = filter(lambda x: x[1] == 0, clusters)
        return [empty_cluster[0] for empty_cluster in empty_clusters]

    def get_farthest_points(self, X):
        def distance_sum(Xi, centers):
            return sum(self.distance_fn(Xi, center) for center in centers)

        non_empty_centers = map(lambda x: x[1], filter(lambda x: x[0] not in self.empty_cluster_idxs, enumerate(self.cluster_centers)))
        return max(map(lambda x: [x, distance_sum(x, non_empty_centers)], X), key=lambda x: x[1])[0]

    def process_empty_clusters(self, X):
        for empty_cluster_idx in self.empty_cluster_idxs:
            cur_center = self.get_farthest_points(X)
            while any(self.cmp_arr(cur_center, center) for center in self.cluster_centers):
                cur_center = self.get_farthest_points(X)
                self.cluster_centers[empty_cluster_idx] = cur_center
        return self.cluster_centers

    def get_cluster_centers(self, X, y):
        ret = [[0 for _ in range(self.n_features)] for _ in range(self.n_clusters)]
        for Xi, center_num in zip(X, y):
            for j in range(self.n_features):
                ret[center_num][j] += Xi[j] / self.cluster_samples_cnt[center_num]
        return ret

    def fit(self, X):
        self.n_features = len(X[0])
        self.cluster_centers = self.init_cluster_centers(X)
        for i in range(self.max_iter):
            while True:
                y = self.get_nearest_centers(X)
                self.cluster_samples_cnt = Counter(y)
                self.empty_cluster_idxs = self.get_empty_cluster_idxs()
                if self.empty_cluster_idxs:
                    self.cluster_centers = self.process_empty_clusters(X)
                else:
                    break
            new_centers = self.get_cluster_centers(X, y)
            self.cluster_centers = new_centers
            print("Iteration:%d" % i)

    def _predict(self, Xi):
        return self.get_nearest_center(Xi)

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


def min_max_scale(X):
    m = len(X[0])
    x_max = [-float('inf') for _ in range(m)]
    x_min = [float('inf') for _ in range(m)]
    for row in X:
        x_max = [max(a, b) for a, b in zip(x_max, row)]
        x_min = [min(a, b) for a, b in zip(x_min, row)]
    ret = []
    for row in X:
        tmp = [(x-b)/(a-b) for a, b, x in zip(x_max, x_min, row)]
        ret.append(tmp)
    return ret


if __name__ == '__main__':
    X, y = load_data("./data/breast_cancer.csv")
    X = min_max_scale(X)
    est = KMeans(n_clusters=2)
    k = 2
    est.fit(X)
    prob_pos = sum(y) / len(y)
    print("Positive probability of X is:%.1f%%.\n" % (prob_pos * 100))
    y_hat = est.predict(X)
    cluster_pos_tot_cnt = {i: [0, 0] for i in range(k)}
    for yi_hat, yi in zip(y_hat, y):
        cluster_pos_tot_cnt[yi_hat][0] += yi
        cluster_pos_tot_cnt[yi_hat][1] += 1
    cluster_prob_pos = {k: v[0] / v[1] for k, v in cluster_pos_tot_cnt.items()}
    for i in range(k):
        tot_cnt = cluster_pos_tot_cnt[i][1]
        prob_pos = cluster_prob_pos[i]
        print("Count of elements in cluster %d is:%d." %
              (i, tot_cnt))
        print("Positive probability of cluster %d is:%.1f%%.\n" % (i, prob_pos * 100))