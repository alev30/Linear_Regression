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


"""classify low mpg as 1, mid as 2, high as 3, for better graphical 
representation of data"""

for row in dataset:
    if row[0] <= buckets[0]:
        row[0] = 1
    elif row[0] <= buckets[1]:
        row[0] = 2
    else:
        row[0] = 3

## problem 2 in another file

## train / test split ##
X_list_train = X_list[0:200]
X_list_test = X_list[200:]

Y_list_train = Y_list[0:200]
Y_list_test = Y_list[200:]

X_train = np.array(X_list_train)
X_test = np.array(X_list_test)

Y_train = np.array(Y_list_train)
Y_test = np.array(Y_list_test)

X = np.array(X_list)
Y = np.array(Y_list)
## feature scaling ##

#colmeans = []
#colstd = []
#for i in range(0,7):
#    colmeans.append(X[:,i].mean())
#    colstd.append(X[:,i].std())
#    X[:,i] = X[:,i] - colmeans[i]
#    X[:,i] = X[:,i] / colstd[i]
#    X_test[:,i] = X_test[:,i] - colmeans[i]
#    X_test[:,i] = X_test[:,i] / colstd[i]       
#    X_train[:,i] = X_train[:,i] - colmeans[i]
#    X_train[:,i] = X_train[:,i] / colstd[i]
#    
#Y_mean = Y.mean()
#Y_std = Y.std()
#Y = Y - Y_mean
#Y = Y / Y_std
#Y_test = Y_test - Y_mean
#Y_test = Y_test / Y_std
#Y_train = Y_train - Y_mean
#Y_train = Y_train / Y_std

## solver ##

## helper function to compute values for polynomial calculations

def polyhelper(value, order):
    table = [1]
    x = value
    for i in range(1,order+1):
        table.append(x)
        x = x*value
    return table

def modifiedpolyhelper(valuematrix,order):
    table = [1]
    for value in valuematrix:
        x = value
        for i in range(1,order+1):
            table.append(x)
            x = x*value
    return table

def polytable(polymat, matrix, polyval, listrep = False):
    output= matrix
    if (listrep== True):
        for row in polymat:
            matrix.append(modifiedpolyhelper(row,polyval))
    else:
        for value in polymat:
            matrix.append(polyhelper(value,polyval))
    matrix.pop(0)
    return output

def weighthelper(X,Y):
    xTx = np.matmul(X.T,X)
    xTx_inv = np.linalg.pinv(xTx)
    xTy = np.matmul(X.T,Y)
    W = np.matmul(xTx_inv, xTy)
    return W

def error_calc(X,Y,W, size):
    error= 0
    for i in range(0,size):
        iterable  = 0
        for j in range(0, len(W)):
            iterable += X[i][j]*W[j]
        error += (Y[i] - iterable)**2
    error = error/ size
    return error

class solver:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        
    ## single variable polynomial calculator 
    def single_var_poly(self, polyVal, index):
        self.index = index
        self.polyVal = polyVal
        ## generate X table of values given a certain order to go to
        x_poly_matrix = [[]]    
        x_col = self.X[:,index].tolist()
        x_poly_matrix = polytable(x_col,x_poly_matrix, polyVal)    
        newX = np.array(x_poly_matrix)
        self.XpolyTable = newX
        self.W = weighthelper(newX,self.Y)
    
    ## train error squared given M samples
    def train_error_calc(self, multi = False):
        self.trainerror = 0
        if (multi == False):
            self.trainerror = error_calc(self.XpolyTable,self.Y,self.W,200)
        else:
            self.trainerror = error_calc(self.Xmulti_polyTable, self.Y,
                                         self.W,200)
    ## test error squared calculation given M samples
    def test_error_calc(self, xtest, ytest,multi= False):
        self.xtest = xtest
        self.ytest = ytest
        x_test_poly_matrix = [[]]
        if (multi == False):
            x_col = xtest[:,self.index].tolist()
            x_test_poly_matrix =  polytable(x_col,x_test_poly_matrix,
                                            self.polyVal)
            newX = np.array(x_test_poly_matrix)
            self.xtest_polytable = newX
            self.testerror = error_calc(self.xtest_polytable,self.ytest,
                                        self.W,192)
        else:
            x_test_poly_matrix = polytable(self.xtest,x_test_poly_matrix,
                                           self.multi_polyVal, True)
            newX = np.array(x_test_poly_matrix)
            self.xtest_polytable = newX
            self.testerror = error_calc(self.xtest_polytable,self.ytest,
                                        self.W,192)

    ## modified for multivariable, with new modified polyhelper func
    def multivarpoly(self, poly):
        self.multi_polyVal = poly
        ## generate X table of values given a certain order to go to
        x_poly_matrix = [[]]
        x_poly_matrix =  polytable(self.X,x_poly_matrix, poly, True)
        newX = np.array(x_poly_matrix)
        self.Xmulti_polyTable = newX
        self.W = weighthelper(newX, self.Y)
    

###################end of solver object#######################################        


weightset = [[solver(X_train, Y_train) for i in range(0,4)] for j in range (0,7)]


test = [[]]
train = [[]]
weightlist = [[]]
functions = []
rangesplot = []
rangesminplot = []
for i in range(0,7):
    rangesplot.append(np.amax(X[:,i]))
    rangesminplot.append(np.amin(X[:,i]))
    temp1 = []
    temp2 = []
    for j in range(0,4):
        weightset[i][j].single_var_poly(j,i)
        weightset[i][j].train_error_calc()
        weightset[i][j].test_error_calc(X_test,Y_test)
        weightlist.append(weightset[i][j].W)
        temp1.append(weightset[i][j].trainerror)
        temp2.append(weightset[i][j].testerror)
    
    train.append(temp1)
    test.append(temp2)
        
weightlist.pop(0)
train.pop(0)
test.pop(0)

train_error_4 = pd.DataFrame(train, index = ['cylinders', 'displacement','horsepower',
                                         'weight', 'acceleration', 'model year',
                                         'origin'], columns = ['0th order',
                                                 '1st order', '2nd order',
                                                 '3rd order'] )
test_error_4 = pd.DataFrame(test, index = ['cylinders', 'displacement','horsepower',
                                         'weight', 'acceleration', 'model year',
                                         'origin'], columns = ['0th order',
                                                 '1st order', '2nd order',
                                                 '3rd order'] )


## graphing the specific coefficients applied to polynomial values
## scatter plot the test values versus the coefficints providing the lines
## through training



scatterval = 0
plotval = 0
for i in range(0,28,4):
    x = np.linspace(rangesminplot[plotval],rangesplot[plotval],1000)

    plt.scatter(X_test[:,scatterval],Y_test)
    #plt.scatter(X_train[:,scatterval],Y_train)
    plt.plot(x, weightlist[i][0]*(x**0), label = 'intercept')
    plt.plot(x, weightlist[i+1][0]*(x**0) + weightlist[i+1][1]*(x**1), label = 'linear')
    plt.plot(x, weightlist[i+2][0]*(x**0) + weightlist[i+2][1]*(x**1) +
             weightlist[i+2][2]*(x**2), label = 'quadratic')
    plt.plot(x, weightlist[i+3][0]*(x**0) + weightlist[i+3][1]*(x**1) +
             weightlist[i+3][2]*(x**2) +
             weightlist[i+3][3]*(x**3) , label = 'cubic')
    plt.xlabel('Values for feature %s' %feat_names[scatterval+1])
    plt.ylabel('MPG')
    plotval= plotval + 1
    scatterval = scatterval + 1
    plt.show()

multi_feature_poly = solver(X_train,Y_train)
multi_feature_poly.multivarpoly(0)
multi_feature_poly.test_error_calc(X_test,Y_test,True)
multi_feature_poly.train_error_calc(True)
mfp_error = [[multi_feature_poly.trainerror],[multi_feature_poly.testerror]]

multi_feature_poly.multivarpoly(1)
multi_feature_poly.test_error_calc(X_test,Y_test,True)
multi_feature_poly.train_error_calc(True)
mfp_error[0].append(multi_feature_poly.trainerror)
mfp_error[1].append(multi_feature_poly.testerror)

multi_feature_poly.multivarpoly(2)
multi_feature_poly.test_error_calc(X_test,Y_test,True)
multi_feature_poly.train_error_calc(True)
mfp_error[0].append(multi_feature_poly.trainerror)
mfp_error[1].append(multi_feature_poly.testerror)

mfp_error_table = pd.DataFrame(mfp_error, index = ['Training MSE','Testing MSE'],
                               columns = ['0th Order','1st Order', '2nd Order'])

## number 6, logistic regression using sklearn and precision using sklearn

temp = np.array(dataset[0:200])
temp2 = np.array(dataset[200:])
Y_classify= temp [:,0]
Y_true = temp2[:,0]
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_classify)


Y_pred = classifier.predict(X_test)
Y_pred_train = classifier.predict(X_train)
from sklearn.metrics import precision_score
prec_score_test = precision_score(Y_true,Y_pred,average= 'macro')
prec_score_train = precision_score(Y_classify, Y_pred_train, average = 'macro')

## number 7, using the models to predict new values ##

origin_1 =[6,350,180,3700,9,80,1]
regressor = solver(X_train,Y_train)
regressor.multivarpoly(2)
origin_poly = modifiedpolyhelper(origin_1,2)
mpg_pred = np.matmul(regressor.W,origin_poly)

predvar = [origin_1]
mpg_class_pred= classifier.predict(predvar)
