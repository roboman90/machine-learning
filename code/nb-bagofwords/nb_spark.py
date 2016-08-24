##### spark fit #####
from pyspark import SparkContext
from pyspark.sql import SparkSession
from nltk.stem.porter import PorterStemmer


sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'
filenames_read= dataPath+'Books_5_processed_words.json/part-*'

data = sc.textFile(filenames_read)
subset = data.sample(withReplacement=False, fraction=1, seed=5).cache()
train, test = subset.randomSplit([9999, 1])


import json
from pyspark.sql import Row
porter = PorterStemmer()
def convertToDF(line):
    line = json.loads(line)
    words = line['reviewText']
    stemmed = []
    for w in words:
        w = porter.stem(w)
        stemmed.append(w)
    return Row(words=stemmed, label=int(line['classLabel']))


# use spark ML package
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

wordsDataTrain_rdd = train.map(convertToDF)
wordsDataTrain = spark.createDataFrame(wordsDataTrain_rdd)
wordsDataTest_rdd = test.map(convertToDF)
wordsDataTest = spark.createDataFrame(wordsDataTest_rdd)
tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
tfDataTrain = tf.transform(wordsDataTrain)
tfDataTest = tf.transform(wordsDataTest)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tfDataTrain)
rescaledData_train = idfModel.transform(tfDataTrain)
rescaledData_test = idfModel.transform(tfDataTest)

from pyspark.ml.classification import NaiveBayes
# naive bayes (multinomial)
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(rescaledData_train)
result = model.transform(rescaledData_test)

def unpack(line):
    return line.prediction == line.label

comparison = result.select('prediction', 'label').rdd.map(unpack)
comparison_true = comparison.filter(lambda x: x == True)

n1 = comparison_true.count()
n2 = comparison.count()
print('Multinomial Naive Bayes Accuracy Score: ' + str(float(n1)/n2))

