import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## preprocessing

with open('auto-mpg.data') as datafile:
     dataset = [[value for value in line.split()] for line in datafile]
      
carnames = [[]]
Y_list = [[]]
X_list = [[]]
bucketlist = [[]]
feat_names = ['mpg', 'cylinders', 'displacement','horsepower', 
               'weight', 'acceleration', 'model year', 'origin']

from random import shuffle
shuffle(dataset)

for row in dataset:
    for col in range (0,8):
        row[col] = float(row[col])
    carnames.append(row[8:])
    del row[8:]
    Y_list.append(row[0])
    bucketlist.append(row[0])
    X_list.append(row[1:])
    
carnames.pop(0)
Y_list.pop(0)
X_list.pop(0)
bucketlist.pop(0)
bucketlist.sort()
buckets = [bucketlist[130],bucketlist[260]]

## pairwise plots
import seaborn as sns

"""classify low mpg as 1, mid as 2, high as 3, for better graphical 
representation of data"""

for row in dataset:
    if row[0] <= buckets[0]:
        row[0] = 1
    elif row[0] <= buckets[1]:
        row[0] = 2
    else:
        row[0] = 3



dFrame = pd.DataFrame(dataset, columns = ['mpg', 'cylinders', 'displacement','horsepower',
                                         'weight', 'acceleration', 'model year',
                                         'origin'] )

g = sns.pairplot(dFrame, hue = 'mpg')
