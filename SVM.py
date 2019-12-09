from sklearn.svm import SVC
from prepare_data import prepareData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from statistics import mean

def svm(word_frequency, C, kernel, poly_degree):
    train_X, train_Y, test_X, test_Y = prepareData(word_frequency)

    clf = SVC(C=C, kernel=kernel, degree=poly_degree)

    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)

    acc = accuracy_score(test_Y, pred)
    '''
    for i in range(100):
        print(test_Y[i], list(pred)[i])
    '''
    return acc

def grid_search_word_frequency():
    word_frequencys = [100, 200, 300, 400, 500, 1000, 1500]
    score = []
    for word_frequency in word_frequencys:
        accuracys = []
        for i in range(3):
            accuracy = svm(word_frequency, 1, 'linear', 3)
            accuracys.append(accuracy)
        mean_accuracy = mean(accuracys)
        score.append([word_frequency, mean_accuracy])

    print(score)

grid_search_word_frequency()
