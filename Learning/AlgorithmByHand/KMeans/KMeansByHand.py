# -*- coding: utf-8 -*-
# @time       : 4/11/2019 8:52 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : KMeansByHand.py
# @description:
from random import seed, randint, random
from collections import Counter
from sklearn import cluster

from Learning.AlgorithmByHand.Tool.Distance import get_euclidean_distance, get_cosine_distance
from Learning.AlgorithmByHand.Preprocess import load_data, min_max_scale, train_test_split
from Learning.AlgorithmByHand import DataPath
from Learning.AlgorithmByHand.Evaulte import run_time, get_accuracy, get_precision, get_recall


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
        # 初始化质心
        self.cluster_centers = self.init_cluster_centers(X)
        for i in range(self.max_iter):
            while True:
                # 根据质心划分簇
                y = self.get_nearest_centers(X)
                # 统计各簇的样本
                self.cluster_samples_cnt = Counter(y)
                # 处理空簇
                self.empty_cluster_idxs = self.get_empty_cluster_idxs()
                if self.empty_cluster_idxs:
                    self.cluster_centers = self.process_empty_clusters(X)
                else:
                    break
            # 更新质心
            new_centers = self.get_cluster_centers(X, y)
            self.cluster_centers = new_centers
            print("Iteration:%d" % i)

    def _predict(self, Xi):
        return self.get_nearest_center(Xi)

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


def model_evaluation(y, y_pred):
    print("Accuracy:", get_accuracy(y, y_pred))
    print("Precision:", get_precision(y, y_pred))
    print("Recall:", get_recall(y, y_pred))


def main():
    X, y = load_data(DataPath.BREAST_CANCER)
    X = min_max_scale(X)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    @run_time
    def kmeans():
        print("Testing the performance of KMeans ...")
        est = KMeans(n_clusters=2)
        est.fit(X_train)
        y_pred = est.predict(X_test)
        model_evaluation(y_test, y_pred)

    @run_time
    def _sklearn():
        print("Testing the performance of KMeans(sklearn) ...")
        est = cluster.KMeans(n_clusters=2)
        est.fit(X_train)
        y_pred = est.predict(X_test)
        model_evaluation(y_test, y_pred)


main()
