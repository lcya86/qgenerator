from sklearn import tree
from CreateDataset import creatDataSet
import numpy as np
from sklearn import preprocessing
from Common import analysisParagraph


clf = tree.DecisionTreeClassifier()
le = preprocessing.LabelEncoder()


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


if __name__ == '__main__':
    dataset = np.array(creatDataSet())
    print(dataset)

    # dataset[:,3] = le.fit_transform(dataset[:,3])
    dataset[:, 6] = le.fit_transform(dataset[:, 6])
    dataset[:, 7] = le.fit_transform(dataset[:, 7])

    print('training.....')
    print('\n')
    clf = clf.fit(dataset[:, 4:], dataset[:, 2])

    classify('We are a long way from conclusion on North Korea, maybe things will work out, and maybe they won’t - only time will tell....But the work I am doing now should have been done a long time ago!')
