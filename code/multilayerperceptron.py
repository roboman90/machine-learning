from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()
dataPath = '/home/bowen/Documents/data/'

# 1. generate sample dataset using sklearn
from sklearn.datasets import make_classification
import math
import json

n_samples = 10000
n_features = 50
n_classes = 10
n_clusters_per_class= 3
n_informative= int(math.log(n_classes*n_clusters_per_class+1, 2))+1
xs, ys = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, n_informative=n_informative, random_state=5)

filename= dataPath+'classification_data_%s_features_%s_labels_%s_samples.txt' % (str(n_features), str(n_classes), str(n_samples))
f = open(filename, 'w')
for i in xrange(0, n_samples):
    vec = list(xs[i])
    label = ys[i]
    line = {'label': label,
            'features': vec}
    line = json.dumps(line) +'\n'
    f.write(line)
f.close()

############ Spark ML Lib ############

# 2. Load training data
print '>>>>> loading data '
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
data_rdd = sc.textFile(filename)
def convertToDataFrame(line):
    line = json.loads(line)
    return Row(label=int(line['label']), features=Vectors.dense(line['features']))
data_rdd = data_rdd.map(lambda x: convertToDataFrame(x)).cache()
data_df = spark.createDataFrame(data_rdd)
# Split the data into train and test
splits = data_df.randomSplit([0.6, 0.4])
train = splits[0]
test = splits[1]

# 3. training
print '>>>>> training  '
from pyspark.ml.classification import MultilayerPerceptronClassifier
mlp = MultilayerPerceptronClassifier(maxIter=1000, tol=1e-4, seed=1, layers=[n_features, n_features, n_features, n_features, n_classes],
                                     blockSize=100, stepSize=0.03, solver="l-bfgs", initialWeights=None)
model = mlp.fit(train)

# 4. compute accuracy on the test set
print '>>>>> testing  '
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Spark Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))



