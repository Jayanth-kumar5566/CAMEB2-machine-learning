#!/usr/bin/env python3

'''
Run this code as 

	./machine_learning.py input_data y_train y_test dimension_reduction ml_algo param_dimred param_ml out
	
input_data			: The input dataset
y_train				: The training class labels
y_test				: The testing class labels
dimension_reduction	: "none", "pca", "VAE", "rp"
ml_algo 			: "logit", "rf" 
param_dimred		: parameters for dimension reduction algorthim seperated by ,
	"none"	- none
	"pca"	- ratio(explanined variance to preserve)
	"rp"	- epsilon value
	"vae"	- dimensions 80,20
param_ml			: parameters for ML algorithm
	"none"	- none  [all built in] #just incase
out					: output directory
'''

def plot_pca(pca,out):
    y=pca.explained_variance_ratio_
    x= ["C"+str(i) for i in range(1,len(y)+1)]
    plt.bar(x,y)
    plt.ylabel("Explained Variance ratio")
    plt.savefig(out+"explained_variance.png",dpi=600)
    plt.close()
    y_cum=np.cumsum(y)
    plt.plot(y_cum)
    plt.ylabel("Cumulative Explained Variance")
    plt.savefig(out+"Cumulative_exp_variance.png",dpi=600)
    
def rp(X_train,X_test,ep):
    # GRP
    rf = GaussianRandomProjection(eps=ep)
    rf.fit(X_train)
    # applying GRP to the whole training and the test set.
    X_train = rf.transform(X_train)
    X_test = rf.transform(X_test)
    return(X_train,X_test)  
    
    
def pca(X_train,X_test,ratio,out):
    pca = PCA()
    pca.fit(X_train)
    n_comp = 0
    ratio_sum = 0.0
    for comp in pca.explained_variance_ratio_:
        ratio_sum += comp
        n_comp += 1
        if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
            break
    pca = PCA(n_components=n_comp)
    pca.fit(X_train)
    plot_pca(pca,out)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return(X_train,X_test)
    
#Dimension reduction --VAE to 14_dimension
def saveLossProgress_ylim(history):
        loss_collector = []
        loss_max_atTheEnd = 0.0
        for hist in history.history:
            current = history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_atTheEnd:
                loss_max_atTheEnd = current[-1]
        return loss_collector, loss_max_atTheEnd


def saveLossProgress(history,out):
        loss_collector, loss_max_atTheEnd = saveLossProgress_ylim(history)
        figureName = str(out)+"vae_plot.png"
        plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'],loc='upper right')
        plt.savefig(figureName)
        plt.close()
        if 'recon_loss' in history.history:
            figureName = 'reconstruction_loss_detailed'
            plt.ylim(min(loss_collector) * 0.9, loss_max_atTheEnd * 2.0)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['recon_loss'])
            plt.plot(history.history['val_recon_loss'])
            plt.plot(history.history['kl_loss'])
            plt.plot(history.history['val_kl_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss', 'recon_loss', 'val recon_loss', 'kl_loss', 'val kl_loss'], loc='upper right')
            plt.savefig(str(out)+figureName + '.png')
            plt.close()

def vae(X_train,y_train,X_test,dims = [14], epochs=2000, batch_size=1, verbose=2, loss='mse', output_act=False, act='relu', patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, no_trn=False,seed=0):
        # callbacks for each epoch
        modelName = "vae_model" + '.h5'
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]
        # warm-up callback
        warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])  # , print(epoch), print(K.get_value(beta))])
        # warm-up implementation
        def warm_up(epoch):
            val = epoch * warmup_rate
            if val <= 1.0:
                K.set_value(beta, val)
        # add warm-up callback if requested
        if warmup:
            beta = K.variable(value=0.0)
            callbacks.append(warm_up_cb)
        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X_train,y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=seed,
                                                                                    stratify=y_train)
        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])
        # create vae model
        print(dims)
        vae, encoder, decoder = DNN_models.variational_AE(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)
        vae.summary()
        encoder.summary()
        decoder.summary()
        if no_trn:
                return
        # fit
        history = vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))
        # load best model
        vae.load_weights(modelName)
        encoder = vae.layers[1]
        # applying the learned encoder into the whole training and the test set.
        X_train_m, _, X_train = encoder.predict(X_train) #mean, variance,sample
        X_test_m, _, X_test = encoder.predict(X_test) #mean, variance,sample
        return(X_train,X_test,history)
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle

args=sys.argv

#importing the dataset
data=pd.read_csv(args[1],index_col=0)
y_train=pd.read_csv(args[2],index_col=0)
y_test=pd.read_csv(args[3],index_col=0)

#Splitting train test
X_train=data.reindex(y_train.index)
X_test=data.reindex(y_test.index)
scaler = StandardScaler()
scaler.fit(data)
y_train=y_train["ExacerbatorState"]
y_train=y_train.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
y_test=y_test["ExacerbatorState"]
y_test=y_test.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})

#Z-normalisation
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Dimension reduction
if args[4]=="none":
	X_train_d,X_test_d=X_train,X_test	
elif args[4]=="pca":
	X_train_d,X_test_d = pca(X_train,X_test,ratio=float(args[6]),out=args[8])
elif args[4]=="rp":
	X_train_d,X_test_d = rp(X_train,X_test,ep=float(args[6]))
elif args[4]=="vae":
	X_train_d,X_test_d,history= vae(X_train,y_train,X_test,dims=[int(i) for i in args[6].split(",")])
	saveLossProgress(history,args[8])
else:
	sys.exit("check input parameters -- Dimension reduction")

print("Dimension reduction --- Completed")
print(X_train_d.shape)

#Machine learning
scoring={"Acc":make_scorer(balanced_accuracy_score),"F1":make_scorer(f1_score),"F2":make_scorer(fbeta_score,beta=2)}    
if args[5]=="rf":
	hyper_parameters = [{'n_estimators': [150],'criterion':['gini'],
		                    'max_features': ['auto'],
		                    'max_depth':[None],
		                    'min_samples_split':[i for i in np.linspace(0.001,0.5,50)],
		                    'min_samples_leaf':[i for i in np.linspace(0.001,0.5,50)],
		                    }, ]
	clf = GridSearchCV(RandomForestClassifier(class_weight="balanced",bootstrap=True), hyper_parameters, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), scoring=scoring, n_jobs=-1, verbose=1,refit=False,return_train_score=True)
	
elif args[5]=="logit":
	# define models and parameters
	solvers = ['liblinear']
	penalty = ['l1']
	c_values = [1000,100, 10, 1.0, 0.1, 0.01,0.001] 
	# define grid search
	grid = dict(solver=solvers,penalty=penalty,C=c_values)
	model = LogisticRegression(fit_intercept=True,class_weight="balanced")
	clf = GridSearchCV(estimator=model, param_grid=grid, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), scoring=scoring,n_jobs=-1,return_train_score=True,refit=False)
	
clf.fit(X_train_d, y_train)                         
results = clf.cv_results_
df=pd.DataFrame(results)
df_sel=df.loc[:,["params","mean_train_Acc","mean_train_F1","mean_train_F2","mean_test_Acc","mean_test_F1","mean_test_F2"]]
df_sel.to_csv(str(args[8])+"params_acc.csv")
#print params of the maximum mean_test_F2
m_ind=df_sel["mean_test_F2"].idxmax()
print("Best params based on mean test F2 score")
print(df_sel.loc[m_ind,"params"])
print("Maximum F2 score")
print(df_sel.loc[m_ind,["mean_train_Acc","mean_train_F1","mean_train_F2","mean_test_Acc","mean_test_F1","mean_test_F2"]])
