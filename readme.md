###### Ordering Events based on Prediction

To build model for ordering events starting from word embedding(Word 2 vec) to classification Algorithms.
In this code I have used gensim word embedding on our corpus and also used glove word embedding. I used svm,
random forest classifier, Decision tree classification to train the model and compared them. The accuracy is around 42%
for word embedding based on our corpus and svm, with glove word embedding the accuracy is around 45%.

To install the required packages, run the following command
`pip3 install -r requirements.txt`

The repo contains data folder for data to train and test model. Alongside this the baseline.py file is for word embedding
learn from our corpus and classifier_comparision.py to compare diff algos on Glove word embedding.

NOTE:: To RUN this code you will need to download both GLove and nltk.
