from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/data/'

# 1. generate dataset
from sklearn.datasets.samples_generator import make_blobs
import json

n_samples= 10000
n_features=2
n_centers=4
max_iter=100
xs, ys = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

filename= dataPath+'clustering_data_%s_features_%s_centers_%s_samples.txt' % (str(n_features), str(n_centers), str(n_samples))
f = open(filename, 'w')
for i in xrange(0, n_samples):
    vec = list(xs[i])
    line = {'features': vec}
    line = json.dumps(line) +'\n'
    f.write(line)
f.close()

############ Spark ML Lib ############
print '================ spark ML Kmeans =================='

# 2. Load training data
print '>>>>> loading data '
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
data = sc.textFile(filename)
def convertToDataFrame(line):
    line = json.loads(line)
    return Row(features=Vectors.dense(line['features']))
def convertToList(line):
    line = json.loads(line)
    line = line['features']
    return line
data_rdd = data.map(lambda x: convertToDataFrame(x)).cache()
data_df = spark.createDataFrame(data_rdd)
data_list = data.map(lambda x: convertToList(x)).cache()

# 3. training

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
print '>>>>> training '
kmeans = KMeans(maxIter=max_iter).setK(n_centers).setSeed(1)
model = kmeans.fit(data_df)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
print '>>>>> evaluate '
wcsse = model.computeCost(data_df)
print("Within Set Sum of Squared Errors = " + str(wcsse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


############ Sklearn fit ############
print '================ sklearn Kmeans =================='
from sklearn.cluster import KMeans
km = KMeans(n_clusters=n_centers, init='k-means++', n_init=10, max_iter=max_iter, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
km.fit(xs)

print 'Within Set Sum of Squared Errors = ' + str(km.inertia_)
print 'Cluster Centers:', km.cluster_centers_

############ Sklearn partial fit #############
print '================ sklearn Kmeans partial fit =================='

from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=n_centers, init='k-means++', max_iter=max_iter, batch_size=100, verbose=0, compute_labels=True,
                     random_state=None, tol=0.0, max_no_improvement=100, init_size=None, n_init=3, reassignment_ratio=0.01)

km.partial_fit(xs)
print '))))))))))))))))))))))))', km.inertia_

tmp = [0.1]*10
splits = data_list.randomSplit(tmp, 1)
errorTot = 0
for split in splits:
    split = split.collect()
    km.partial_fit(split)
    errorTot+=km.inertia_
    del split

print 'Within Set Sum of Squared Errors = ' + str(errorTot)
print 'Cluster Centers:', km.cluster_centers_


