##### data processing #####
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
import json


sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'ratings_books_small.csv'

# read the file
# user, item, rating, timestamp
data = sc.textFile(filename_read)

data = data.map(lambda x: x.split(','))
uniqueItems = data.map(lambda x: x[1]).distinct().collect()
print 'Number of Items: ', len(uniqueItems)
n = len(uniqueItems)

def createSparseVector(line, totalLength, uniqueItems):
    indices = []
    values = []
    line = json.loads(line)
    user = line[0]
    reviews = line[1]
    for row in reviews:
        item = row[1]
        rating = int(row[2])
        indices.append(uniqueItems.index(item))
        values.append(rating)
    Row(user=user, features=Vectors.sparse(totalLength, indices, values))

data_rdd = data.map(lambda line: createSparseVector(line, n, uniqueItems))
data_df = spark.createDataFrame(data_rdd).cache()

