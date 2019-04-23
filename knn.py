import random
import numpy as np
from sklearn import datasets
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, LeaveOneOut

def seq_deletion(knn,X,y):
    x_copy = X.tolist()
    y_copy = y.tolist()
    i = 0
    for x_i,y_i in zip(X,y):
        x_copy = x_copy[:i]+x_copy[i+1:]
        y_copy = y_copy[:i]+y_copy[i+1:]
        knn.fit(x_copy,y_copy)
        p = knn.predict([x_i])
        if p[0] != y_i:
            x_copy = x_copy[:i]+[x_i]+x_copy[i:]
            y_copy = y_copy[:i]+[y_i]+y_copy[i:]
            i += 1
        return x_copy,y_copy

def seq_insertion(knn,X,y):
    x_copy = X[:knn.n_neighbors].tolist()
    y_copy = y[:knn.n_neighbors].tolist()
    for x_i,y_i in zip(X,y):
        knn.fit(x_copy,y_copy)
        p = knn.predict([x_i])
        if p[0] != y_i:
            x_copy.append(x_i)
            y_copy.append(y_i)
    return x_copy,y_copy

def evaluate(knn,v,selection=None):
    score = []
    for train_index, test_index in v:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if selection != None:
            X_train, y_train = selection(knn,X_train,y_train)
        knn.fit(X_train, y_train)
        score.append(knn.score(X_test, y_test))

    mean = np.mean(score)
    std = np.std(score)
    return mean,std

def bootstrap(n,n_bootstraps=10,n_train=0.9):
    n_train = int(n_train*n)
    output = []
    for _ in range(n_bootstraps):
        train_idxs = [random.randint(0,n-1) for _ in range(n_train)]
        test_idxs = [i for i in range(n) if not i in train_idxs]
        output.append((train_idxs,test_idxs))
    return output

def run_knn(metrics,ks,selection=None):
    for k in ks:
        for metric in metrics:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            
            bs = bootstrap(len(x),n_bootstraps=10,n_train=0.9)
            score = evaluate(knn,bs,selection)
            print('Bootstrap',k,metric,score)

            kf = KFold(n_splits=10)
            score = evaluate(knn,kf.split(x),selection)
            print('KFold',k,metric,score)

            loo = LeaveOneOut()
            score = evaluate(knn,loo.split(x),selection)
            print('LeaveOneOut',k,metric,score)

iris = datasets.load_iris()

x = iris.data
y = iris.target

def seuclidean_3(x,y):
    r = np.sqrt(np.sum((x - y)**2 / 3))
    return r

'''
Primeiro slide: variar k e variar amostragem
'''
metrics = ['euclidean']
ks = [1,2,3,4,5]
run_knn(metrics,ks)

'''
Segundo slide: para um dado k: variar medidas de distância
e comparar com algoritmos iterativos
'''
metrics = ['euclidean', 'manhattan', 'canberra', 'braycurtis', seuclidean_3]
ks = [3]
run_knn(metrics,ks)

metrics = ['euclidean']
print('seq_deletion')
run_knn(metrics,ks,seq_deletion)

print('seq_insertion')
run_knn(metrics,ks,seq_insertion)