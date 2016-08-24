##### sklearn fit #####
from pyspark import SparkContext
from pyspark.sql import SparkSession
from nltk.stem.porter import PorterStemmer

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'
filenames_read= dataPath+'Books_5_processed_words.json/part-*'

# read all the files generated previously
data = sc.textFile(filenames_read)

porter = PorterStemmer()
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


####### local learning (fit) #######
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

# let's load a smaller subset to save memory and time
subset = data.sample(withReplacement=False, fraction=0.001, seed=5).cache()

xs = subset.map(data_process_xs).collect()
ys = subset.map(data_process_ys).collect()

counter = CountVectorizer()
bagOfWords = counter.fit_transform(xs)

tfidf = TfidfTransformer()
bagOfWordsFrequency = tfidf.fit_transform(bagOfWords).toarray()

bagOfWordsFrequency_train, bagOfWordsFrequency_test, ys_train, ys_test = train_test_split(bagOfWordsFrequency, ys, test_size=0.3)

## Naive Bayes Classifiers ##
# 1. Gaussian Naive Bayes (fit)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(bagOfWordsFrequency_train, ys_train)
ys_predict = gaussian.predict(bagOfWordsFrequency_test)

from sklearn.metrics import accuracy_score
print 'Gaussian Naive Bayes Accuracy Score: ', accuracy_score(ys_test, ys_predict)

# 2. Multinomial Naive Bayes (fit)

from sklearn.naive_bayes import MultinomialNB
multinomial = MultinomialNB()
multinomial.fit(bagOfWordsFrequency_train, ys_train)
ys_predict = multinomial.predict(bagOfWordsFrequency_test)
print 'Multinomial Naive Bayes Accuracy Score: ', accuracy_score(ys_test, ys_predict)



