##### data processing #####
from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'Books_5.json'

# read the file
data = sc.textFile(filename_read)
# take only the reviews text data, and the
def convertStringToUnicode(s):
    u = unicode(s, "utf-8")
    return u

import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = stopwords.words('english')

def dataProcessing(line, threshold):
    line = json.loads(line)
    # obtain the text
    text = line[convertStringToUnicode('reviewText')]
    # tokenize
    tokenizedText = []
    for x in text.split(' '):
        if x.lower() not in stops:
            tokenizedText.append(x)
    # convert label to integers as class labels
    rating = int(line[convertStringToUnicode('overall')])
    # generate class label (positive vs negative reviews)
    if rating >= threshold:
        sentiment = 1 # positive
    else:
        sentiment = 0 # negative
    newLine = {
        'reviewText': tokenizedText,
        'rating': rating,
        'classLabel': sentiment
    }
    newLine = json.dumps(newLine)
    return newLine

# randomly sample a fraction of the original dataset
data = data.sample(withReplacement=False, fraction=1, seed=5)
# call the dataProcessing function on each line of data
newData = data.map(lambda x: dataProcessing(x, threshold=4))
# write to file
filename_write= dataPath+'Books_5_processed_words.json'
newData.saveAsTextFile(filename_write)

