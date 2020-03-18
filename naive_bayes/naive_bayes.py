# -*- coding: utf-8 -*-
# @Time    : 2020/3/15 下午7:39
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : naive_bayes.py
# @Desc    : 朴素贝叶斯算法实现

import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha  # 贝叶斯系数
        self.y_count = None  # Y每种类型的数量
        self.y_proba = None  # Y每种类型的概率
        self.x_proba = {}   # (xi编号, xi取值, y取值): 概率

    def fit(self, x_train, y_train):
        y_train = pd.DataFrame(y_train)
        self.y_count = y_train[0].value_counts()
        self.y_proba = (self.y_count + self.alpha) / (len(y_train) + len(self.y_count) * self.alpha)

        x_train = pd.DataFrame(x_train)
        for c in x_train.columns:
            for j in self.y_proba.index:
                t = x_train[y_train[0] == j][c].value_counts()
                for i in t.index:
                    self.x_proba[(c, i, j)] = (t[i] + self.alpha) / (self.y_count[j] + len(t) * self.alpha)

    def predict(self, x_test):
        result = []
        for y in self.y_proba.index:
            p = self.y_proba[y]
            for i, x in enumerate(x_test):
                p *= self.x_proba[(i, x, y)]
            print('{} 对应概率: {}'.format(y, p))
            result.append(p)
        return self.y_proba.index[np.argmax(result)]


def main():
    x_train = [
        [1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
        [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
        [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']
    ]
    y_train = [
        -1, -1, 1, 1, -1,
        -1, -1, 1, 1, 1,
        1, 1, 1, 1, -1
    ]

    model = NaiveBayes(alpha=0.2)
    model.fit(x_train, y_train)

    x_test = [2, 'S']
    y_pred = model.predict(x_test)
    print('分类结果：', y_pred)


if __name__ == '__main__':
    main()
