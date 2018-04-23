import sys
import getopt
import re
import csv
from collections import namedtuple
from Common import analysisParagraph

def initModel():
    # input questions data
    questions = []
    with open('data/questions.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        Row = namedtuple('Row', headers)
        for r in f_csv:
            row = Row(*r)
            questions.append(row)

    return questions


def satisfyRanage(number, range_str):
    reg = re.compile(r'[-+]?[0-9]*\.?[0-9]+')
    numbers = reg.findall(range_str)
    if range_str.startswith('(') and number <= float(numbers[0]):
        return False
    elif range_str.startswith('[') and number < float(numbers[0]):
        return False
    if range_str.endswith(')') and number >= float(numbers[1]):
        return False
    elif range_str.endswith(']') and number > float(numbers[1]):
        return False
    return True


def generateQuestion(keyword, polarity, subjectivity, tense, pos_tag):
    questions = initModel()
    resultQuestions = []
    for item in questions:
        if satisfyRanage(polarity, item[1]) and satisfyRanage(subjectivity, item[2]) and (tense in item[3].split(',') or item[3] == 'ALL') and (pos_tag in item[4].split(',') or item[4] == 'ALL'):
            resultQuestions.append(item[0])

    return resultQuestions


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h")
    except getopt.GetoptError:
        print('genstion.py <paragraph>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('genstion.py <paragraph>')
            sys.exit()

    if(len(argv) == 1):
        analysisResult = analysisParagraph(argv[0])
        print('\n')
        print(generateQuestion(
            analysisResult[0], analysisResult[1], analysisResult[2], analysisResult[3], analysisResult[4]))
    else:
        print('<paragraph> is one single string')


if __name__ == "__main__":
    main(sys.argv[1:])
