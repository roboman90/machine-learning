from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/data/'

filename = dataPath + 'clustering_data_2_features_4_centers_100000_samples.txt'

n_samples= 100000
n_features=2
n_centers=4
max_iter=100

print '================ spark ML Kmeans =================='

# 1. Load training data
print '>>>>> loading data '
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import json
data = sc.textFile(filename)
def convertToRow(line):
    line = json.loads(line)
    return Row(features=Vectors.dense(line['features']))
def convertToList(line):
    line = json.loads(line)
    line = line['features']
    return line
data_rdd = data.map(convertToRow).cache()
data_df = spark.createDataFrame(data_rdd).cache()

# 2. training

from pyspark.ml.clustering import KMeans
import datetime
# Trains a k-means model.
kmeans = KMeans(maxIter=max_iter).setK(n_centers).setSeed(1)
start = datetime.datetime.today()
model = kmeans.fit(data_df)
end = datetime.datetime.today()
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wcsse = model.computeCost(data_df)
print("Within Cluster Sum of Squared Errors = " + str(wcsse))

# Shows the result.
centers = model.clusterCenters()
print "Cluster Centers: ", centers

print 'runtime: ', end-start

