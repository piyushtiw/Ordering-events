import numpy as np
import pandas as pd
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RegexpStemmer
from nltk.stem.snowball import SnowballStemmer
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



stop = stopwords.words("english")
st = RegexpStemmer('ing$', min=8)
stemmer = SnowballStemmer("english")
encoding="utf-8"


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
        self.data['Events'] = self.data.apply(lambda row: (row['Event 1'] + " " + row['Event 2']), axis=1)
        self.data['Event 1'] = self.data.apply(lambda row: word_tokenize(row['Event 1']), axis=1)
        self.data['Event 2'] = self.data.apply(lambda row: word_tokenize(row['Event 2']), axis=1)
        self.preprocess('Event 1')
        self.preprocess('Event 2')
        self.preprocess('Events')
        return self.data


if __name__ == '__main__':
    obj = LoadData()
    file = 'data/P1_training_set.csv'
    obj.load_data_file(file)
    train, test = obj.test_train_data()

    obj1 = NormaliseData(train)
    processed_data = obj1.tokenize()

    mult_nb = Pipeline(
        [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    mult_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    svc_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

    all_models = [
        ("mult_nb", mult_nb),
        ("mult_nb_tfidf", mult_nb_tfidf),
        ("bern_nb", bern_nb),
        ("bern_nb_tfidf", bern_nb_tfidf),
        ("svc", svc),
        ("svc_tfidf", svc_tfidf),
    ]

    unsorted_scores = [(name, cross_val_score(model, processed_data['Events'], processed_data['Label'], cv=5).mean()) for name, model in all_models]
    scores = sorted(unsorted_scores, key=lambda x: -x[1])

    print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))