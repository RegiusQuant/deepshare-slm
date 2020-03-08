# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 下午6:38
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : perceptron.py
# @Desc    : 感知机算法实现

import numpy as np


class Perceptron:
    def __init__(self):
        self.w = None  # 模型参数w
        self.b = None  # 模型参数b
        self.lr = 1.0  # 学习率
        self.max_iter = 100  # 最多迭代轮数

    def fit(self, x, y):
        # 初始化模型参数
        self.w = np.zeros(x.shape[1])
        self.b = 0

        for _ in range(self.max_iter):
            for i in range(x.shape[0]):
                # 判断是否误判
                if y[i] * (np.dot(self.w, x[i]) + self.b) <= 0:
                    self.w += self.lr * np.dot(y[i], x[i])
                    self.b += self.lr * y[i]


def main():
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    model = Perceptron()
    model.fit(x, y)
    print('w:', model.w, 'b:', model.b)


if __name__ == '__main__':
    main()
