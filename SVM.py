from sklearn.svm import SVC
from prepare_data import prepareData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def svm():
    train_X, train_Y, test_X, test_Y = prepareData(word_frequency=25)

    clf = SVC(C=1, kernel='linear')
    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)

    acc = accuracy_score(test_Y, pred)
    print(acc)

    for i in range(100):
        print(test_Y[i], list(pred)[i])

def grid_search():
    svm()

grid_search()
