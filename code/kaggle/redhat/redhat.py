##### spark fit #####
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/data/kaggle/redhat/'

filenames_people = dataPath + 'people.json'
filenames_train = dataPath + 'act_train.json'
filenames_test = dataPath + 'act_test.json'



