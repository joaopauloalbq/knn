import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, LeaveOneOut


def seq_deletion(knn, X, y):
    x_copy = X.tolist()
    y_copy = y.tolist()
    for x_i, y_i in zip(X, y):
        x_copy = x_copy[1:]
        y_copy = y_copy[1:]
        knn.fit(x_copy, y_copy)
        p = knn.predict([x_i])
        if p[0] != y_i:
            x_copy.append(x_i)
            y_copy.append(y_i)
    return x_copy, y_copy


def seq_insertion(knn, X, y):
    x_copy = X[:knn.n_neighbors].tolist()
    y_copy = y[:knn.n_neighbors].tolist()
    for x_i, y_i in zip(X, y):
        knn.fit(x_copy, y_copy)
        p = knn.predict([x_i])
        if p[0] != y_i:
            x_copy.append(x_i)
            y_copy.append(y_i)
    return x_copy, y_copy


def evaluate(knn, v, selection=None):
    score = []
    for train_index, test_index in v:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if selection != None:
            X_train, y_train = selection(knn, X_train, y_train)
        knn.fit(X_train, y_train)
        score.append(knn.score(X_test, y_test))
    mean = np.mean(score)
    std = np.std(score)
    return mean, std


def bootstrap(n, n_bootstraps=10, n_train=0.9):
    n_train = int(n_train*n)
    output = []
    for _ in range(n_bootstraps):
        train_idxs = [random.randint(0, n-1) for _ in range(n_train)]
        test_idxs = [i for i in range(n) if not i in train_idxs]
        output.append((train_idxs, test_idxs))
    return output


def run_knn(metrics, ks, selection=None):
    results = dict(k=[], metric=[], method=[], mean=[], std=[])
    for k in ks:
        for metric in metrics:
            results['k'].extend(3*[k])
            results['metric'].extend(3*[str(metric)])
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

            bs = bootstrap(len(x), n_bootstraps=10, n_train=0.9)
            score = evaluate(knn, bs, selection)
            results['method'].append('Bootstrap')
            results['mean'].append(score[0])
            results['std'].append(score[1])
            print('Bootstrap', k, metric, score)

            kf = KFold(n_splits=10)
            score = evaluate(knn, kf.split(x), selection)
            results['method'].append('KFold')
            results['mean'].append(score[0])
            results['std'].append(score[1])
            print('KFold', k, metric, score)

            loo = LeaveOneOut()
            score = evaluate(knn, loo.split(x), selection)
            results['method'].append('LeaveOneOut')
            results['mean'].append(score[0])
            results['std'].append(score[1])
            print('LeaveOneOut', k, metric, score)
    return pd.DataFrame.from_dict(results)


def plot_results(results, type_):
    assert type_ in ['slide1', 'slide2', 'deletion', 'insertion']

    if type_ == 'slide1':
        groups = results.groupby(by=['method', 'metric'])
    elif type_ == 'slide2':
        groups = results.groupby(by=['method', 'k'])
    else:
        groups = results.groupby(by=['metric', 'k'])

    for group in groups.groups:
        attr1, attr2 = group
        idx = groups.groups[group]
        data = results.iloc[idx]
        ax = np.arange(len(idx))

        plt.errorbar(ax, data['mean'], data['std'],
                     linestyle='None', marker='o')

        if type_ == 'slide1':
            plt.xticks(ax, data['k'])
            plt.xlabel('k')
            plt.title('{}; Métrica de distância: {}'.format(attr1, attr2))
        elif type_ == 'slide2':
            plt.xticks(ax, data['metric'])
            plt.xlabel('Métrica de distância')
            plt.title('{}; k = {}'.format(attr1, attr2))
        else:
            plt.xticks(ax, data['method'])
            plt.xlabel('Método de amostragem')
            plt.title('Métrica de distância: {}; k = {}'.format(attr1, attr2))
        plt.ylabel('Performance')
        plt.savefig('{}-{}-{}.pdf'.format(attr1, attr2, type_), format='pdf')
        plt.close()


iris = datasets.load_iris()

x = iris.data
y = iris.target


'''
Primeiro slide: variar k e variar amostragem
'''
metrics = ['euclidean']
ks = [1, 2, 3, 4, 5]
results = run_knn(metrics, ks)
plot_results(results, 'slide1')

'''
Segundo slide: para um dado k: variar medidas de distância
e comparar com algoritmos iterativos
'''
metrics = ['euclidean', 'manhattan', 'canberra', 'braycurtis']
ks = [3]
results = run_knn(metrics, ks)
plot_results(results, 'slide2')

metrics = ['euclidean']
print('seq_deletion')
results_deletion = run_knn(metrics, ks, seq_deletion)
plot_results(results_deletion, 'deletion')

print('seq_insertion')
results_insertion = run_knn(metrics, ks, seq_insertion)
plot_results(results_insertion, 'insertion')
