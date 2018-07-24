from __future__ import print_function
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import argparse
import os.path as osp
import scipy.sparse as sp
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn.feature_selection

dataset = pd.read_csv('maline_fvt2.csv')

#print(dataset.head)


X = dataset.iloc[:, 1:157].values
y = dataset.iloc[:,0].values

#print(X)


##normalize

scaler = MinMaxScaler()
scaler.fit(X)
MinMaxScaler(copy=True, feature_range=(0, 1))

X_normalized = scaler.transform(X)

print(X_normalized.shape)



##feature selection

sel = SelectKBest(chi2, k='all')


print(sel.fit_transform(X_normalized, y))




#sel.transform(X)


#print(sel.scores_)

#np.savetxt("deb.csv", sel.scores_, delimiter=",")



####creating the table

l=list(dataset.columns.values)

l=l[1:157]
score=sel.scores_
df = pd.DataFrame({'system_call': pd.Series(l, dtype=str), 'chi_score': pd.Series(score, dtype=float)})
columnsTitles=['system_call','chi_score']
df=df.reindex(columns=columnsTitles)

df['Feature_Rank'] = df['chi_score'].rank(ascending=False)
df=df.sort_values(by=['Feature_Rank'])
df.to_csv("score4.csv",index=False)


#print(sel.scores_)
#np.savetxt("foo.csv", sel.scores_, delimiter=",")



'''

seed = 7
num_trees = 100
max_features = 3
test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X_transform, y, test_size=test_size,random_state=seed)

kfold = KFold(n_splits=10, random_state=7)
#model = RandomForestClassifier(n_estimators=num_trees)

model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

model.fit(X_train, Y_train)
#********************* compute Classification Accuracy in train*********************
print("******************************")
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
print(("The accuracy of Classification in train: %.3f (%.3f)") % (results.mean(), results.std()))
#********************* compute Classification Accuracy in test*********************
print("******************************")
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X_test,Y_test, cv=kfold, scoring=scoring)
print(("The accuracy of Classification in test 1: %.3f (%.3f)") % (results.mean(), results.std()))
#********************* compute Classification Accuracy in train*********************
#print("******************************")
predictions = model.predict(X_test)
#print("The accuracy of Classification in test:")
#print(accuracy_score(Y_test, predictions))
print("******************************")
print("confusion_matrix:")
print(confusion_matrix(Y_test, predictions))
#******** precision , recall, f1-score, support****************************
print("******************************")
print("classification_report:")
print(classification_report(Y_test, predictions))
#********************* compute Logarithmic Loss in Train****************************
print("******************************")
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
print(("The Loss of Classification in train data: %.3f (%.3f)") % (results.mean(), results.std()))
#********************* compute Logarithmic Loss in Test**************************** 
print("******************************")
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X_test,Y_test, cv=kfold, scoring=scoring)
print(("The Loss of Classification in test data:: %.3f (%.3f)") % (results.mean(), results.std()))
#********************* compute Area Under ROC Curve in Train*************************
print("******************************")
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
print(("The Area Under ROC Curve in Train: %.3f (%.3f)") % (results.mean(), results.std()))
#********************* compute Area Under ROC Curve in Test*************************
print("******************************")
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X_test,Y_test, cv=kfold, scoring=scoring)
print(("The Area Under ROC Curve in test: %.3f (%.3f)") % (results.mean(), results.std()))
#*****************************Compute FPR and TPR**************************
print("******************************")
FPR, TPR, thresholds = metrics.roc_curve(Y_test, predictions, pos_label=2)
print("The FPR result:")
print(FPR)
#*****************************End of Compute Metrics***********************

'''

