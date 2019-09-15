import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import scale
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression

class MeanEmbeddingTransformer(TransformerMixin):

    def __init__(self):
        self._vocab, self._E = self._load_words()

    def _load_words(self):
        E = {}
        vocab = []

        with open('data/glove.6B.100d.txt', 'r',
                  encoding="utf8") as file:
            for i, line in enumerate(file):
                l = line.split(' ')
                if l[0].isalpha():
                    v = [float(i) for i in l[1:]]
                    E[l[0]] = np.array(v)
                    vocab.append(l[0])
        return np.array(vocab), E

    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None

    def _doc_mean(self, doc):
        return np.mean(np.array([self._E[w.lower().strip()] for w in doc if w.lower().strip() in self._E]), axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def print_scores(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred, pos_label='positive', average='micro')))

def tokenize_and_transform(X, sample_size):
    events1 = X[:, 0]
    events2 = X[:, 1]
    tok_e1 = [word_tokenize(doc) for doc in events1[:sample_size]]
    tok_e2 = [word_tokenize(doc) for doc in events2[:sample_size]]
    met = MeanEmbeddingTransformer()
    X_transform = np.append(met.fit_transform(tok_e1), met.fit_transform(tok_e2), axis=1)
    return X_transform

if __name__ == '__main__':
    train_data = pd.read_csv('data/P1_training_set.csv', sep=',')
    test_data = pd.read_csv('data/P1_testing_set.csv', sep=',')

    data = train_data

    X_train = data[['Event 1', 'Event 2']].as_matrix()
    y_train = data['Label'].as_matrix()

    X_train_transform = tokenize_and_transform(X_train, 160000)
    np.savetxt('glove_word_2_vec.csv', X_train_transform, delimiter=',')

    X_train_transform = np.loadtxt('glove_word_2_vec.csv', delimiter=',')
    X_train_transform = scale(X_train_transform)

    rus = RandomUnderSampler(random_state=0)
    X_resample, y_resample = rus.fit_sample(X_train_transform, y_train[:X_train_transform.shape[0]])

    X_train, X_test, y_train, y_test = train_test_split(X_resample,
                                                        y_resample, stratify=y_resample, random_state=0)

    lr = LogisticRegression()
    print_scores(lr, X_train, y_train, X_test, y_test)

    svc = SVC().fit(X_train, y_train)
    print_scores(svc, X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier().fit(X_train, y_train)
    print_scores(rf, X_train, y_train, X_test, y_test)

    dtc = DecisionTreeClassifier().fit(X_train, y_train)
    print_scores(dtc, X_train, y_train, X_test, y_test)








