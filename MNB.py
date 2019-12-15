from sklearn.naive_bayes import MultinomialNB
from prepare_data import prepareData
from sklearn.metrics import accuracy_score


def mnb():
    train_X, train_Y, test_X, test_Y = prepareData(word_frequency=5)

    clf = MultinomialNB()
    clf.fit(train_X, train_Y)

    train_pred = clf.predict(train_X)
    test_pred = clf.predict(test_X)

    train_acc = accuracy_score(train_Y, train_pred)
    test_acc = accuracy_score(test_Y, test_pred)

    print('Train Acc: ', train_acc)
    print('Test Acc: ', test_acc)


def grid_search():
    mnb()

grid_search()