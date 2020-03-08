# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 下午7:03
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : perceptron_sklearn.py
# @Desc    : 感知机算法实现(sklearn)


import numpy as np
from sklearn.linear_model import Perceptron


def main():
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    model = Perceptron()
    model.fit(x, y)
    print('w:', model.coef_, 'b:', model.intercept_, 'num_iter:', model.n_iter_)
    print('Accuracy:', model.score(x, y))


if __name__ == '__main__':
    main()
