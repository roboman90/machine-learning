from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
import json

spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'ratings_Books.csv'

# load data
data = sc.textFile(filename_read)

# formatting
def f(x):
    x = x.split(',')
    return (x[0], [(x[1], x[2])])

data1 = data.map(f)

# reduce (groupby)
def f2(list1, line):
    return list1 + line


data2 = data1.reduceByKey(f2) # original approach

# jsonify and write to file
def jsonify(x):
    return json.dumps(x)+'\n'

filename_write = dataPath+'ratings_Books_groupedby_user.json'

data3 = data2.map(jsonify)

data3.saveAsTextFile(filename_write)



