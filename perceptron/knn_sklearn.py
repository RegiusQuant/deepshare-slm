# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 下午10:06
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : knn_sklearn.py
# @Desc    : K近邻算法实现(sklearn)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    x_train = np.array([[5, 4], [9, 6], [4, 7], [2, 3], [8, 1], [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    x_test = np.array([[5, 3]])

    for k in range(1, 6, 2):
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print('K: {}  Predict: {}'.format(k, y_pred))


if __name__ == '__main__':
    main()
