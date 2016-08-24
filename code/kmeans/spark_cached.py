from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/data/'

# 1. generate dataset
from sklearn.datasets.samples_generator import make_blobs
import json
count = 1000
for i in xrange(0, count):
    n_samples= 1000000
    n_features=20*10
    n_centers=500
    max_iter=100
    xs, ys = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

    filename= dataPath+'clustering_data_%s_features_%s_centers_%s_samples.txt' % (str(n_features), str(n_centers), str(n_samples*count))
    f = open(filename, 'a')
    for i in xrange(0, n_samples):
        vec = list(xs[i])
        line = {'features': vec}
        line = json.dumps(line) +'\n'
        f.write(line)
    f.close()


print '================ spark ML Kmeans =================='

# 2. Load training data
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import json
data = sc.textFile(filename)
def convertToDataFrame(line):
    line = json.loads(line)
    return Row(features=Vectors.dense(line['features']))
def convertToList(line):
    line = json.loads(line)
    line = line['features']
    return line
data_rdd = data.map(lambda x: convertToDataFrame(x)).cache()
data_df = spark.createDataFrame(data_rdd).cache()

# 3. training

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

