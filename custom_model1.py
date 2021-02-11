#Implementing random forests with feature engineering

# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skbio.stats.composition import clr

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
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


#Reading the data - training
df=pd.read_csv("./../Data_21Dec20/species_data.csv",index_col=0)
y_train=pd.read_csv("./../METADATA/data_194.csv",index_col=0)
y_test=pd.read_csv("./../METADATA/data_test.csv",index_col=0)

# Feature-selection LEFSe
f=pd.read_csv("./data/feature_sel_LEFSe/selected_microbes_95.csv",index_col=0).index
df_sel=df.loc[:,f]
del df,f


#Data transformation - CLR
df_norm=clr(df_sel+1)
df_norm=pd.DataFrame(df_norm,index=df_sel.index,columns=df_sel.columns)
del df_sel
'''
#Data transformation - Relative_abundace
df_norm=df_sel.div(df_sel.sum(axis=1),axis=0)*100
'''

#Pathway dataset
df=pd.read_csv("./../MASTER-TABLES/HUMANN2/humann2_unipathway_pathabundance_cpm.tsv",sep='\t',index_col=0)
pathways=[i.find("|")==-1 for i in df.index]
df=df.loc[pathways,:]
del pathways

df.columns=[i.split("_")[0] for i in df.columns]
df.drop(["13LTBlank","76LTBlank","Blank"],axis=1,inplace=True)
df.drop(["UNMAPPED","UNINTEGRATED"],axis=0,inplace=True)

df["Super_pathway"]=[i.split(";")[0] for i in df.index]
df=df.groupby("Super_pathway")
df=df.sum()

df=df.transpose()

# Feature-selection LEFSe
f=pd.read_csv("./data/feature_sel_LEFSe/pathways/selected_pathways_3class_90.tsv",sep='\t',index_col=0).index
df_sel=df.loc[df_norm.index,f]
del df,f

#Pathway_norm
pdf_norm=clr(df_sel+1)
pdf_norm=pd.DataFrame(pdf_norm,index=df_sel.index,columns=df_sel.columns)
del df_sel

#Merge dataframes
data=pd.merge(df_norm,pdf_norm,left_index=True,right_index=True)

#Training and testing splitting
X_train_d=data.reindex(y_train.index)
X_test_d=data.reindex(y_test.index)
del data
#y_test and train - class replacment
y_train=y_train["ExacerbatorState"]
y_train=y_train.replace({"NonEx":0,"Exacerbator":1,"FreqEx":2})
y_test=y_test["ExacerbatorState"]
y_test=y_test.replace({"NonEx":0,"Exacerbator":1,"FreqEx":2})


#Dimension reduction --VAE to 14_dimension
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
        vae, encoder, decoder = DNN_models.variational_AE(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)
        vae.summary()
        if no_trn:
                return
        # fit
        history = vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))
        # load best model
        vae.load_weights(modelName)
        encoder = vae.layers[1]
        # applying the learned encoder into the whole training and the test set.
        X_train_m, _, X_train = encoder.predict(X_train)
        X_test_m, _, X_test = encoder.predict(X_test)
        return(X_train_m,X_test_m,history)

#Dimensionality reduction
D_X_train_d,D_X_test_d,history= vae(X_train_d,y_train,X_test_d)


#Logistic-regression
lr = LogisticRegression(random_state=0,penalty="l1",class_weight="balanced",n_jobs=-1)
lr.fit(D_X_train_d,y_train)
print("Training Acc",lr.score(D_X_train_d,y_train))
print("Testing Acc",lr.score(D_X_test_d,y_test))
y_pred=lr.predict(D_X_test_d)
print("Confusion Matrix ",confusion_matrix(y_test,y_pred))
print("F Score in weighted fashion ",f1_score(y_test,y_pred,average="weighted"))


#Random Forest
hyper_parameters = [{'n_estimators': [150],'criterion':['gini'],
                        'max_features': ['auto'],
                        'max_depth':[s for s in range(2, 20, 2)],
                        'min_samples_split':[s for s in range(2, 20, 2)],
                        'min_samples_leaf':[s for s in np.arange(0.001, 0.1, 0.005)],
                        }, ]
scoring={"Acc":make_scorer(accuracy_score)}

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced",bootstrap=True), hyper_parameters, cv=LeaveOneOut(), scoring=scoring, n_jobs=-1, verbose=1,refit="Acc",return_train_score=True)

clf.fit(X_train_d, y_train)                         

results = clf.cv_results_
df=pd.DataFrame(results)

df_test=df.filter(regex=("split.*_test_Acc"))

def score(x,y=y_train):
	'''
	x is the cv result row
	y is y_train
	'''
	x=x.reset_index(drop=True)
	y=y.reset_index(drop=True)
	z=pd.DataFrame(x).join(y)
	zg=z.groupby("ExacerbatorState")
	table=zg.mean()
	balanced_acc=np.mean(table)
	return(balanced_acc.values[0])

Acuracy_cbalanced=df_test.apply(score,axis=1)
Acuracy_cbalanced=pd.DataFrame(Acuracy_cbalanced,columns=["Class_average Accuracy"])
del df_test


#===================Training_Acc============================
train_acc_balanced=df["mean_train_Acc"]


print("Class balanced Accuracy",Acuracy_cbalanced)
print("Max Class balanced Accuracy",Acuracy_cbalanced.max())

def get_nest(df):
    return(df["n_estimators"])

def get_maxdept(df):
    return(df["max_depth"])
    
def get_min_ss(df):
    return(df["min_samples_split"])
    
def get_min_sl(df):
    return(df["min_samples_leaf"])    

Acuracy_cbalanced["n_estimators"]=df.loc[:,"params"].apply(get_nest)
Acuracy_cbalanced["max_depth"]=df.loc[:,"params"].apply(get_maxdept)
Acuracy_cbalanced["min_samples_split"]=df.loc[:,"params"].apply(get_min_ss)
Acuracy_cbalanced["min_samples_leaf"]=df.loc[:,"params"].apply(get_min_sl)
Acuracy_cbalanced["training_Acc"]=train_acc_balanced

Acuracy_cbalanced.to_csv("tuning_res.csv")

'''
#Input the best parameters and then run
x=[]
for i in range(100):
    rf=RandomForestClassifier(n_jobs=-1, n_estimators=150,min_samples_split=12,max_depth=8,min_samples_leaf
=0.001,class_weight="balanced",bootstrap=True)
    rf.fit(X_train_d, y_train)                         
    y_pred=rf.predict(X_test_d)
    print(confusion_matrix(y_test,y_pred))
    x.append(rf.score(X_test_d,y_test))

print("Median of testing acc",np.median(x))
'''
