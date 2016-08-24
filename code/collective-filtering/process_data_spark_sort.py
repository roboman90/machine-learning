from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
import json
import datetime

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

# sort by key
data2 = data1.sortByKey()

# save to disk
filename_write = dataPath+'ratings_Books_sortedby_user.json'
# jsonify and write to file
def jsonify(x):
    line = {'user': x[0],
            'pair': x[1]}
    return json.dumps(line)


data3 = data2.map(jsonify)
data3.coalesce(1).saveAsTextFile(filename_write)

# load the data and do the groupby
f = open(filename_write+'/part-00000', 'r')
filename_write_grouped = dataPath+'ratings_Books_sortedby_user_grouped.json'
f2 = open(filename_write_grouped, 'w')
user = ''
pairs = []
for row in f:
    row = json.loads(row)

    if user == '':
        user = row['user']
        pairs.append(row['pair'])
    else:
        if user == row['user']:
            pairs.append(row['pair'])
        else:
            line = [user, pairs]
            line = json.dumps(line)+'\n'
            f2.write(line)
            user = row['user']
            pairs=[row['pair']]
f.close()
f2.close()

print('Runtime Sort Only:')
print(datetime.datetime.today()-start)
