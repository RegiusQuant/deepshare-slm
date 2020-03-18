# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 下午9:28
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : knn.py
# @Desc    : K近邻算法实现

import numpy as np


class KNN:
    def __init__(self, x_train, y_train, k):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, x_test):
        dist = [(np.linalg.norm(x_test - x_i, ord=2), y_i)
                for x_i, y_i in zip(self.x_train, self.y_train)]
        dist.sort()
        candidates = [d[1] for d in dist[:self.k]]
        return max(candidates, key=candidates.count)


def main():
    x_train = np.array([[5, 4], [9, 6], [4, 7], [2, 3], [8, 1], [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    x_test = np.array([5, 3])

    for k in range(1, 6, 2):
        model = KNN(x_train, y_train, k)
        y_pred = model.predict(x_test)
        print('K: {}  Predict: {}'.format(k, y_pred))


if __name__ == '__main__':
    main()
