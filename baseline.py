import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RegexpStemmer
from nltk.stem.snowball import SnowballStemmer
import string
import gensim
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm


stop = stopwords.words("english")
st = RegexpStemmer('ing$', min=8)
stemmer = SnowballStemmer("english")
encoding="utf-8"
tfidf = TfidfTransformer(norm="l2")
seperator = ' '


class LoadData():
    def __init__(self):
        pass

    def load_data_file(self, file):
        self.data = pd.read_csv(file)

    def test_train_data(self):
        msk = np.random.rand(len(self.data)) < 1.0
        train = self.data[msk]
        test = self.data[~msk]
        return train, test


class NormaliseData():
    def __init__(self, train):
        self.data = train
        self.number_of_events = np.shape(self.data)[0]

    def preprocess(self, column):
        self.data[column] = self.data[column].apply(lambda x: [i for i in x if not i.isdigit()])
        self.data[column] = self.data[column].apply(lambda x: [item for item in x if item not in string.punctuation])
        self.data[column] = self.data[column].apply(lambda x: [item for item in x if item not in stop])
        self.data[column] = self.data[column].apply(lambda x: [stemmer.stem(y) for y in x])

    def tokenize(self):
        self.data['Event 1'] = self.data.apply(lambda row: word_tokenize(row['Event 1']), axis=1)
        self.data['Event 2'] = self.data.apply(lambda row: word_tokenize(row['Event 2']), axis=1)
        self.preprocess('Event 1')
        self.preprocess('Event 2')
        self.data['Events'] = self.data.apply(lambda row: (row['Event 1'] + row['Event 2']), axis=1)
        return self.data


class WordVectors():
    def __init__(self, data):
        self.data = data

    def word_to_vec(self, column):
        return gensim.models.Word2Vec(self.data[column], min_count=1, size=100, window=5)


class MeanEmbeddingVectorizer():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class SvmClassification():
    def __init__(self):
        self.clf = None

    def train(self, wor_to_vec):
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(wor_to_vec, processed_data['Label'])
        self.clf = clf

    def predict(self, event):
        return self.clf.predict(event)



if __name__ == '__main__':
    obj = LoadData()
    file = 'data/P1_training_set.csv'
    obj.load_data_file(file)
    train, test = obj.test_train_data()

    obj1 = NormaliseData(train)
    processed_data = obj1.tokenize()

    obj2 = WordVectors(processed_data)
    event1_vec = obj2.word_to_vec('Event 1')
    event1_w2v = dict(zip(event1_vec.wv.index2word, event1_vec.wv.vectors))
    event2_vec = obj2.word_to_vec('Event 2')
    event2_w2v = dict(zip(event2_vec.wv.index2word, event2_vec.wv.vectors))

    obj4 = MeanEmbeddingVectorizer(event1_w2v)
    meanEvent1Vec = obj4.transform(processed_data['Event 1'])
    obj4 = MeanEmbeddingVectorizer(event2_w2v)
    meanEvent2Vec = obj4.transform(processed_data['Event 2'])

    concate_w2v = meanEvent1Vec + meanEvent2Vec
    np.savetxt('gensim_word_2_vec.csv', concate_w2v, delimiter=',')

    obj5 = SvmClassification()
    vec = np.loadtxt('gensim_word_2_vec.csv', delimiter=',')
    obj5.train(vec)

    file = 'data/P1_testing_set.csv'
    obj.load_data_file(file)
    test, train = obj.test_train_data()

    obj1 = NormaliseData(test)
    test_data = obj1.tokenize()

    right = 0
    wrong = 0

    obj2 = WordVectors(test_data)
    event1_vec = obj2.word_to_vec('Event 1')
    event1_w2v = dict(zip(event1_vec.wv.index2word, event1_vec.wv.vectors))
    event2_vec = obj2.word_to_vec('Event 2')
    event2_w2v = dict(zip(event2_vec.wv.index2word, event2_vec.wv.vectors))

    obj4 = MeanEmbeddingVectorizer(event1_w2v)
    meanEvent1Vec = obj4.transform(test_data['Event 1'])
    obj4 = MeanEmbeddingVectorizer(event2_w2v)
    meanEvent2Vec = obj4.transform(test_data['Event 2'])

    concate_w2v = meanEvent1Vec + meanEvent2Vec

    i = 0
    while i < len(concate_w2v):
        predict = obj5.predict(concate_w2v[i].reshape(1, -1))
        if predict == test_data['Label'][i]:
            right = right + 1
        else:
            wrong = wrong + 1
        i += 1

    print(right/(right+wrong))