#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle


# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# importing keras
'''
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model
'''
# importing util libraries
import datetime
import time
import math
import os
import importlib
'''
# importing custom library
import DNN_models
import exception_handle
'''
args=sys.argv
print(args)


'''
Run this code as 

	./ml_tuned.py input_data ml_alg

input data - is the obj object saved by the ML script	

ml_alg - Machine learning algorithm
	logit - for logistic regression
'''	

fileObj = open(str(args[1]), 'rb')
[X_train_d,X_test_d,y_train,y_test]=pickle.load(fileObj)
fileObj.close()

if args[2]=="logit":
	model = LogisticRegression(fit_intercept=True,class_weight="balanced",solver="liblinear",penalty='l1',C=0.06, n_jobs=-1)
	model.fit(X_train_d, y_train)
	print("Training Acc",model.score(X_train_d,y_train))
	test_score=model.score(X_test_d,y_test)
	print("Testing Acc",test_score)
	y_pred=model.predict(X_test_d)
	print("Confusion Matrix ",confusion_matrix(y_test,y_pred))
	print("F1 Score ",f1_score(y_test,y_pred))
	print("F2 Score ",fbeta_score(y_test,y_pred,beta=2))
	print("Balanced Accuracy ",balanced_accuracy_score(y_test,y_pred))
	
	







