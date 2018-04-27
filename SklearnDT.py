import sys
import getopt
from sklearn import tree
from CreateDataset import creatDataSet
import numpy as np
from sklearn import preprocessing
from Common import analysisParagraph
import csv
from collections import namedtuple


clf = tree.DecisionTreeClassifier()
le = preprocessing.LabelEncoder()

def getQuestionText(index):
    # input questions data
    questions = []
    with open('data/questions.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        Row = namedtuple('Row', headers)
        for r in f_csv:
            row = Row(*r)
            questions.append(row)

    return questions[index-2].question


def classify(text):
    result = clf.predict([
        [
            analysisParagraph(text)[1],
            analysisParagraph(text)[2],
            le.transform([analysisParagraph(text)[3]])[0],
            le.transform([analysisParagraph(text)[4]])[0]
        ]
    ])
    print(result)
    print(getQuestionText(int(result[0])))


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h")
    except getopt.GetoptError:
        print('SklearnDT.py <paragraph>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('SklearnDT.py <paragraph>')
            sys.exit()

    if(len(argv) == 1):
        classify(argv[0])
    else:
        print('<paragraph> is one single string')


if __name__ == "__main__":
    dataset = np.array(creatDataSet())
    # print(dataset)

    # dataset[:,3] = le.fit_transform(dataset[:,3])
    dataset[:, 6] = le.fit_transform(dataset[:, 6])
    dataset[:, 7] = le.fit_transform(dataset[:, 7])

    print('training.....')
    print('\n')
    clf = clf.fit(dataset[:, 4:], dataset[:, 2])
    
    main(sys.argv[1:])