##### sklearn partial fit #####
from pyspark import SparkContext
from pyspark.sql import SparkSession
from nltk.stem.porter import PorterStemmer

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'
filenames_read= dataPath+'Books_5_processed_words.json/part-*'

porter = PorterStemmer()
# read all the files generated previously
data = sc.textFile(filenames_read)
import json
def data_process_xs(line):
    line = json.loads(line)
    words = line['reviewText']
    stemmed = []
    for w in words:
        w = porter.stem(w)
        stemmed.append(w)
    return ' '.join(stemmed)

def data_process_ys(line):
    line = json.loads(line)
    label = line['classLabel']
    return label


from sklearn.cross_validation import train_test_split
import numpy as np

## Naive Bayes Classifiers ##
# 1. Gaussian Naive Bayes (partial fit)
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
from sklearn.naive_bayes import MultinomialNB
multinomial = MultinomialNB()

from sklearn.feature_extraction.text import HashingVectorizer
def tokenizer(x):
    return x.split(' ')

vectorizer = HashingVectorizer(decode_error='ignore',
                               n_features=2**18,
                               non_negative=True,
                               tokenizer=tokenizer)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

subset = data.sample(withReplacement=False, fraction=0.001, seed=5).cache()
numberOfChunks = 10
train, test = subset.randomSplit(weights=[numberOfChunks-1, 1])
splits = train.randomSplit(weights=[1]*(numberOfChunks-1))
for split in splits:
    xs = split.map(data_process_xs).collect()
    ys = split.map(data_process_ys).collect()
    xs = vectorizer.transform(xs).toarray()
    xs = tfidf.fit_transform(xs).toarray()
    gaussian.partial_fit(xs, ys, classes=np.array([0,1]))
    multinomial.partial_fit(xs, ys, classes=np.array([0,1]))
    del xs, ys

xs = test.map(data_process_xs).collect()
ys = test.map(data_process_ys).collect()
xs = vectorizer.transform(xs).toarray()
xs = tfidf.fit_transform(xs).toarray()
ys_predict_g = gaussian.predict(xs)
ys_predict_m = multinomial.predict(xs)

from sklearn.metrics import accuracy_score
print 'Gaussian Naive Bayes (partial fit) Accuracy Score: ', accuracy_score(ys, ys_predict_g)
print 'Multinomial Naive Bayes (partial fit) Accuracy Score: ', accuracy_score(ys, ys_predict_m)


'''
Gaussian Naive Bayes (partial fit) Accuracy Score:  0.670050761421
Multinomial Naive Bayes (partial fit) Accuracy Score:  0.807106598985
'''
