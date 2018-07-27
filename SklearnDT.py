import sys
import imp
sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
import getopt
from sklearn import tree
from CreateDataset import creatDataSet
import numpy as np
from sklearn import preprocessing
from Common import analysisParagraph
import csv
from collections import namedtuple
import nltk


nltk.data.path.append('/tmp')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt',download_dir='/tmp')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords',download_dir='/tmp')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger',download_dir='/tmp')



def predict(text):
    clf = tree.DecisionTreeClassifier()
    le = preprocessing.LabelEncoder()
    dataset = np.array(creatDataSet())
    # print(dataset)

    # dataset[:,3] = le.fit_transform(dataset[:,3])
    dataset[:, 6] = le.fit_transform(dataset[:, 6])
    dataset[:, 7] = le.fit_transform(dataset[:, 7])

    clf = clf.fit(dataset[:, 4:], dataset[:, 2])
    
    return classify(text,clf,le)
    


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


def classify(text,clf,le):
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
    return getQuestionText(int(result[0]))

def lambda_handler(event,context):
    return predict(event['text'])


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
        predict(argv[0])
    else:
        print('<paragraph> is one single string')


if __name__ == "__main__":
    main(sys.argv[1:])