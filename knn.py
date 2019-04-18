from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, KFold, LeaveOneOut


def evaluate(v):
    score = []
    for train_index, test_index in v:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)
        score.append(knn.score(X_test, y_test))

    return score


iris = datasets.load_iris()

x = iris.data
y = iris.target

for k in [1]: #,2,3,4,5
    for metric in ['euclidean']: #, 'manhattan', 'chebyshev', 'minkowski'
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        
        rs = ShuffleSplit(n_splits=10)
        score = evaluate(rs.split(x))
        print(score)

        kf = KFold(n_splits=10)
        score = evaluate(kf.split(x))
        print(score)

        loo = LeaveOneOut()
        score = evaluate(loo.split(x))
        print(score)
