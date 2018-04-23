import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from textblob import TextBlob

def clean(doc):
    stop = set(stopwords.words('english'))
    stop.add("that's")
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return punc_free


def generateTopicWord(cleaned_text):
    dictionary = gensim.corpora.Dictionary(cleaned_text)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_text]

    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=1,
                   id2word=dictionary, passes=50)
    return ldamodel.print_topics(num_topics=1, num_words=1)[0][1].split('*')[1].strip('"')


def analysisParagraph(text):
    text = text.strip('\n')
    doc_complete = sent_tokenize(text)

    # clean text
    doc_clean = [clean(doc).split() for doc in doc_complete]

    # tag pos
    pos_dict = {}
    for sent in doc_clean:
        words = nltk.pos_tag(sent)
        pos_dict.update(dict(words))

    # generate topic word
    topic_word = generateTopicWord(doc_clean)

    # estimate emotion
    blob = TextBlob(text)

    # print debug information
    # print(doc_complete)
    # print(pos_dict)
    # print(topic_word)
    # print(blob.sentiment)
    # print(pos_dict[topic_word])
    # print('tense:'+getTense(pos_dict))
    # print('\n')

    return (topic_word, blob.sentiment.polarity, blob.sentiment.subjectivity, getTense(pos_dict), pos_dict[topic_word])



def getTense(pos_dict):
    if 'VB' in pos_dict.values():
        return 'VB'
    elif 'VBD' in pos_dict.values():
        return 'VBD'
    elif 'VBN' in pos_dict.values():
        return 'VBN'
    elif 'VBG' in pos_dict.values():
        return 'VBG'
    else:
        return 'ALL'
