from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
import json
import datetime
import gc

start = datetime.datetime.today()


spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'ratings_Books.csv'

# load data
data = sc.textFile(filename_read)

# formatting
def f(x):
    x = x.split(',')
    return (x[0], (x[1], x[2]))

data1 = data.map(f)
data2 = data1.sortByKey().cache()
print 'partitions: ', data2.getNumPartitions()

uniqueUsers = data1.map(lambda x: x[0]).distinct()
numOfChunks= 100
chunks_uniqueUsers = uniqueUsers.randomSplit([1]*numOfChunks)

filename_write = dataPath+'ratings_Books_groupedby_user_BruteForce.txt'
f = open(filename_write, 'w')
for chunk in chunks_uniqueUsers:
    users = chunk.collect()
    temp = ''
    for user in users:
        print user
        pairs = data2.lookup(user)
        line = {'user': user, 'pair': pairs}
        temp = temp+json.dumps(line)+'\n'
    f.write(temp)
    del users
    gc.collect()
f.close()

print('Runtime Brute Force:')
print(datetime.datetime.today()-start)
