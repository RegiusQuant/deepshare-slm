# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 下午1:09
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : deepshare-slm
# @File    : naive_bayes_sklearn.py
# @Desc    : 朴素贝叶斯算法实现实现(sklearn)

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB

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

    encoder = OneHotEncoder()
    x_train = encoder.fit_transform(x_train)

    model = MultinomialNB(alpha=1e-4)
    model.fit(x_train, y_train)

    x_test = [[2, 'S']]
    x_test = encoder.transform(x_test)
    y_pred = model.predict(x_test)
    print('分类结果：', y_pred)
    print(model.predict_proba(x_test))


if __name__ == '__main__':
    main()