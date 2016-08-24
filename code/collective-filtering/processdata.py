##### data processing #####
#from pyspark import SparkContext
#from pyspark.sql import SparkSession
#sc = SparkContext()
#spark = SparkSession.builder.getOrCreate()

dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'ratings_Books.csv'

import sqlite3
conn = sqlite3.connect(dataPath + 'ratings_books.db')

c = conn.cursor()

# Create table
try:
    #c.execute('''CREATE TABLE ratings (user text, item text, rating text, timestamp text)''')
    c.execute('''CREATE TABLE users (user text, ratingitempair text)''')
    c.execute('''CREATE TABLE items (item text)''')
except:
    pass

# Insert a row of data
import json
f = open(filename_read, 'r')
n = 1
for line in f:
    print n
    n = n+1

    line = line[0:len(line)-2]
    line = line.split(',')

    user = line[0]
    item = line[1]
    rating = line[2]
    c.execute('''SELECT * FROM users where user='%s' ''' % user)
    val = c.fetchone()
    if val is None:
        c.execute("INSERT INTO users VALUES (?, ?)", (user, json.dumps([])))
    else:
        ratingItemPair = val[1]
        ratingItemPair = json.loads(ratingItemPair)
        ratingItemPair.append((item, rating))
        ratingItemPair = json.dumps(ratingItemPair)
        c.execute("UPDATE users SET ratingitempair = ? WHERE user=? ", (ratingItemPair, user))

    c.execute('''SELECT item FROM items where item='%s' ''' % item)
    val = c.fetchone()
    if val is None:
        c.execute("INSERT INTO items VALUES (?)", (item,))

    conn.commit()

f.close()
# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()


# json format
import json
conn = sqlite3.connect(dataPath + 'ratings_books.db')

c = conn.cursor()
c.execute('''SELECT * FROM users ''')

file_write = dataPath+'ratings_books_aggregated.json'
f = open(file_write, 'w')
for row in c:
    user = row[0]
    c.execute('''SELECT * FROM users WHERE user="%s" ''' % user)
    data = c.fetchone()
    itemRating = json.loads(data[1])
    line = {'user': user,
            'itemrating': itemRating}
    line = json.dumps(line) + '\n'
    f.write(line)
f.close()

conn.close()




######## reduce by key #########
dataPath = '/home/bowen/Documents/amazon-reviews/'

filename_read= dataPath+'ratings_books_small.csv'

data = sc.textFile(filename_read)
def f(x):
    x = x.split(',')
    return (x[0], [(x[1], x[2])])

data1 = data.map(f)

def f2(list1, line):
    return list1 + line

data2 = data1.reduceByKey(f2)

data2.first()


