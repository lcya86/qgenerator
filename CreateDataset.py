import csv
from Common import analysisParagraph
import numpy as np


def creatDataSet():
    dataset = []
    with open('data/labelled_data.csv') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            result = [row['title'], row['message'], row['question_no']]
            result.extend(list(analysisParagraph(
                row['title']+' '+row['message'])))
            dataset.append(result)

    return dataset


def outputDataset(headers, rows):
    with open('data/dataset.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


if __name__ == '__main__':
    dataset = np.array(creatDataSet())
    print(dataset)
    outputDataset(['title', 'message', 'question_no', 'key_word',
                   'emotion', 'subjectivity', 'tense', 'topic'], dataset)
