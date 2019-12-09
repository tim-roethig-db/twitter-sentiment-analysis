from sklearn.svm import SVC
from prepare_data import prepareData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_X, train_Y, test_X, test_Y = prepareData()

def main():
    parameters = {'kernel': ('linear', 'poly', 'rbf'), 'C': [1, 10]}

    clf = SVC(C=1, kernel='linear')
    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)

    acc = accuracy_score(test_Y, pred)
    print(acc)

    print(test_Y)
    print(list(pred))
main()