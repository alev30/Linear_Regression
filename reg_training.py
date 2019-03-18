import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## preprocessing

with open('auto-mpg.data') as datafile:
     dataset = [[value for value in line.split()] for line in datafile]


from random import shuffle
shuffle(dataset)


Y_list = [[]]
X_list = [[]]
feat_names = ['mpg', 'cylinders', 'displacement','horsepower', 
               'weight', 'acceleration', 'model year', 'origin']

from random import shuffle
shuffle(dataset)

for row in dataset:
    for col in range (0,8):
        row[col] = float(row[col])
    Y_list.append(row[0])
    X_list.append(row[1:9])

Y_list.pop(0)
X_list.pop(0)

