import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

testdf = pd.read_csv(r'D:/python/ml/mnist_test.csv')
traindf = pd.read_csv(r'D:/python/ml/mnist_train.csv')

#preprocessing
x_train = traindf.drop('label',axis=1).iloc[0:8000,0:8000]
y_train = traindf['label'].iloc[0:8000]
x_test = testdf.drop('label',axis=1).iloc[0:8000,0:8000]
y_test = testdf['label'].iloc[0:8000]
print(x_train.shape)
print(x_test.shape)
y_test

#test image plotting
plt.figure(figsize=(3,3))
random_num= int(input("enter random number: "))
image = x_train.iloc[random_num].to_numpy().reshape(28,28)
plt.imshow(image, cmap=matplotlib.cm.binary)
print(y_train[random_num])

#Checking Accuracy of ML models
from sklearn.metrics import accuracy_score

#SVM (Linear)
from sklearn.svm import SVC
svml = SVC(kernel="linear")
svml.fit(x_train, y_train)
pred_svml = svml.predict(x_test)
print("Mean squared error svm(linear): ",mean_squared_error(y_test, pred_svml))
print("Accuracy svm(linear): ",accuracy_score(y_test, pred_svml))

#SVM (Polynomial)
svmp = SVC(kernel="poly", degree = 2)
svmp.fit(x_train, y_train)
pred_svmp = svmp.predict(x_test)
print("Mean squared error svm(polynomial): ",mean_squared_error(y_test, pred_svmp))
print("Accuracy svm(polynomial): ",accuracy_score(y_test, pred_svmp))


#SVM (rbf)
svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(x_train, y_train)
pred_svm_rbf = svm_rbf.predict(x_test)
print("Mean squared error svm(rbf): ",mean_squared_error(y_test, pred_svm_rbf))
print("Accuracy svm(rbf): ",accuracy_score(y_test, pred_svm_rbf))

#Logistic Regression
from sklearn.linear_model import LogisticRegression  
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)
print("Mean squared error Logistic Regression: ",mean_squared_error(y_test, pred_lr))
print("Accuracy of Logistic Regression: ", accuracy_score(y_test, pred_lr))


#Naive Bayes
from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB()
nb.fit(x_train, y_train)
pred_nb = nb.predict(x_test)
print("Mean squared error Naive Bayes: ",mean_squared_error(y_test,pred_nb))
print("Accuracy score Naive Bayes: ", accuracy_score(y_test, pred_nb))


#KNN
from sklearn.model_selection import GridSearchCV       #using gridsearchcv to get best value for N hyperparameter tunning
import warnings
from sklearn.neighbors import KNeighborsClassifier
knn_h = KNeighborsClassifier()
knn_h.fit(x_train, y_train)
warnings.filterwarnings('ignore') #to disable printing future warning

parameter_grid = {"n_neighbors" :[2,4,3,7,10,14,11,12,13,15,16,17,18]}
grid = GridSearchCV(knn_h,  parameter_grid, cv = 10, scoring='accuracy',return_train_score = False)
grid.fit(x_train, y_train)

print("Best accuracy found for :",grid.best_params_,"\nAccuracy: ",grid.best_score_)
n = grid.best_params_.get('n_neighbors') #storing best value for n

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
print("Mean squared error KNN: ",mean_squared_error(y_test, pred_knn))
print("Accuracy of KNN: ", accuracy_score(y_test, pred_knn))