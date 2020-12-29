#Implementing random forests with feature engineering

# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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
X_train=pd.read_csv("data/train.csv", index_col=0)
y_train=pd.read_csv("data/train_labels.csv",index_col=0)

#Reading the data - testing
X_test=pd.read_csv("data/test.csv", index_col=0)
y_test=pd.read_csv("data/test_labels.csv",index_col=0)

#Scaling the dataset - Necessary? no different units

#scaler=StandardScaler()
#X=scaler.fit_transform(X) 

#Principal Component Analysis
def pca(X_train, X_test,ratio=0.95,ncomp="Null"): #manuplate the ratio to choose componenets
    # PCA
    pca = PCA()
    pca.fit(X_train)
    n_comp = 0
    ratio_sum = 0.0
    for comp in pca.explained_variance_ratio_:
        ratio_sum += comp
        n_comp += 1 
        if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
            break

    if ncomp=="Null":
    	pca = PCA(n_components=n_comp,whiten=True,svd_solver="full")
    else:
    	pca = PCA(n_components=ncomp,whiten=True,svd_solver="full") #from graph
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    #printing metrics
    print("Explained variance for all comp")
    print(pca.explained_variance_ratio_.sum())
    plt.plot(pca.explained_variance_ratio_,'o-') 
    plt.savefig("plot.png")
    #plt.show()
    # applying the eigenvectors to the whole training and the test set.
    return(X_train,X_test)
    
X_train_d,X_test_d=pca(X_train, X_test)

print("PCA done")

#Gausian Random Projection
def rp(X_train,X_test):
    # GRP
    rf = GaussianRandomProjection(eps=0.5)
    rf.fit(X_train)
    # applying GRP to the whole training and the test set.
    X_train = rf.transform(X_train)
    X_test = rf.transform(X_test)
    return(X_train,X_test)    


'''
#Random forest with leave one out CV
hyper_parameters = [{'n_estimators': [s for s in range(1, 500, 50)],
                        'max_features': ['sqrt'],
                        'max_samples':[s for s in np.arange(0.01,1,0.2)],
                        'max_depth':[None],
                        'min_samples_split':[2]
                        }, ]

scoring={"balanced Acc":make_scorer(balanced_accuracy_score),"mathews corr ceoff":make_scorer(matthews_corrcoef),"Acc":make_scorer(accuracy_score)}

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample",bootstrap=True), hyper_parameters, cv=LeaveOneOut(), scoring=scoring, n_jobs=-1, verbose=1,refit="Acc",return_train_score=True)
clf.fit(X_train_d, y_train.ravel())                         

y_pred=clf.predict(X_test_d)
print(confusion_matrix(y_test,y_pred))


results = clf.cv_results_
df=pd.DataFrame(results)
df.to_csv("Cross_validation_log.csv")

yaxis=["mean_train_mathews corr ceoff","mean_train_balanced Acc","mean_test_mathews corr ceoff","mean_test_balanced Acc","mean_test_Acc","mean_train_Acc"]
xaxis=["param_n_estimators","param_max_samples"]

for x in xaxis:
    plt.clf()
    for y in yaxis:
        plt.plot(df[x],df[y],'-o')
    plt.legend()
    plt.savefig("metrics_plot"+str(x))
    
print("Max LOOCV accuracy is")
print(df.loc[:,"mean_test_Acc"].max())
    
max_index=df.loc[:,"mean_test_Acc"].idxmax()

print(df.loc[max_index,:]["param_n_estimators"])
print(df.loc[max_index,:]["param_max_samples"])
'''


'''
X_train_d,X_test_d=pca(X_train,X_test,ratio=0.8)
# SVM

hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 1)], 'kernel': ['linear']},
                        {'C': [2 ** s for s in np.arange(-7, 3, 0.5)], 'gamma': [2 ** s for s in np.arange(3, -15, -0.5)],'kernel': ['rbf']}]
                        
scoring={"Acc":make_scorer(balanced_accuracy_score)}                            
#weight={0:2,1:1,2:3}
weight="balanced"

clf = GridSearchCV(SVC(probability=False, cache_size=1000,class_weight=weight), hyper_parameters, cv=StratifiedKFold(10), scoring=scoring, n_jobs=-1, verbose=1,refit=False,return_train_score=True)
clf.fit(X_train_d, y_train.ravel())

results = clf.cv_results_
df=pd.DataFrame(results)
df.to_csv("Cross_validation_log.csv")

print("Max LOOCV balanced accuracy is")
print(df.loc[:,"mean_test_Acc"].max())
    
max_index=df.loc[:,"mean_test_Acc"].idxmax()

kernel=df.loc[max_index,:]["param_kernel"]
C=df.loc[max_index,:]["param_C"]
gamma=df.loc[max_index,:]["param_gamma"]

print("Kernel",kernel)
print("gamma",gamma)
print("C",C)

clf=SVC(probability=False, cache_size=1000,class_weight=weight,kernel=kernel,C=C,gamma=gamma)
clf.fit(X_train_d,y_train.ravel())

y_pred=clf.predict(X_test_d)
print(confusion_matrix(y_test,y_pred))
print(confusion_matrix(y_test,y_pred,normalize="true"))
print("Balanced Accuracy", balanced_accuracy_score(y_test,y_pred))


#mask = df.columns.str.contains('split.*_test_Acc')
#df.loc[:,mask]
#df1.sum(axis=1)
'''




#Doesn't learn at-all Class imbalance returns class 1 always
#so, random forest (with PCA) is not good factor this. Probably, needs boosting

'''

# Anamoly detection -- One SVM
#hyper_parameters = [{'kernel': ['linear'],'nu':np.arange(0.1,0.4,0.01)},
#                        {'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf'],'nu':np.arange(0.1,0.4,0.01)}]

res_dic={}
for i in np.arange(0.3,1,0.1):

    X_train_d,X_test_d=pca(X_train, X_test,ratio=i)

    #hyper_parameters = [{'kernel': ['linear'],'nu':np.arange(0.1,0.4,0.01)},
    #                        {'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf'],'nu':np.arange(0.1,0.4,0.01)}]


    hyper_parameters = [{'kernel': ['linear'],'nu':np.arange(0.1,0.5,0.001)},
                            {'gamma': [ s for s in np.linspace(1e-15, 1e-1,100)],'kernel': ['rbf'],'nu':np.arange(0.1,0.5,0.001)}]

    scoring={"Acc":make_scorer(balanced_accuracy_score)}                            
                            
    clf = GridSearchCV(OneClassSVM(cache_size=1000), hyper_parameters, cv=StratifiedKFold(10), scoring=scoring, n_jobs=-1, verbose=1,refit=False,return_train_score=True)
    clf.fit(X_train_d, y_train.ravel())

    results = clf.cv_results_
    df=pd.DataFrame(results)
    df.to_csv("Cross_validation_log.csv")

    print("Max LOOCV accuracy is")
    res_dic[i]=(df.loc[:,"mean_test_Acc"].max())
'''

'''
max_index=df.loc[:,"mean_test_Acc"].idxmax()

kernel=df.loc[max_index,:]["param_kernel"]
gamma=df.loc[max_index,:]["param_gamma"]
nu=df.loc[max_index,:]["param_nu"]

print("Kernel",kernel)
print("gamma",gamma)
print("Nu",nu) 

clf=OneClassSVM(cache_size=1000,kernel=kernel,gamma=gamma,nu=nu)
clf.fit(X_train_d,y_train)

y_pred=clf.predict(X_test_d)
print(confusion_matrix(y_test,y_pred,normalize="all"))
print("Accuracy", accuracy_score(y_test,y_pred))
print("Balanced Accuracy", balanced_accuracy_score(y_test,y_pred))
'''

'''
#Auto-encoders for dimensionality reduction
def ae(X_train,X_test,y_train,seed,dims = [50], epochs= 3000, batch_size=1, verbose=2, loss='mean_squared_error', latent_act=False, output_act=False, act='relu', patience=20, val_rate=0.2, no_trn=False):

    # filename for temporary model checkpoint
    modelName = 'data.h5'

    # clean up model checkpoint before use
    if os.path.isfile(modelName):
        os.remove(modelName)

    # callbacks for each epoch
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                 ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

    # spliting the training set into the inner-train and the inner-test set (validation set)
    #X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X_train, y_train, test_size=val_rate, random_state=seed, stratify=y_train)
    X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X_train, y_train, test_size=val_rate, random_state=seed) #regression

    # insert input shape into dimension list
    dims.insert(0, X_inner_train.shape[1])

    # create autoencoder model
    autoencoder, encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
    autoencoder.summary()

    if no_trn:
        return

    # compile model
    autoencoder.compile(optimizer='adam', loss=loss)

    # fit model
    history = autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size,shuffle=True, callbacks=callbacks,
                         verbose=verbose, validation_data=(X_inner_test, X_inner_test))
    # save loss progress
    saveLossProgress(history,seed)

    # load best model
    autoencoder = load_model(modelName)
    layer_idx = int((len(autoencoder.layers) - 1) / 2)
    encoder = Model(autoencoder.layers[0].input, autoencoder.layers[layer_idx].output)

    # applying the learned encoder into the whole training and the test set.
    X_train = encoder.predict(X_train)
    X_test = encoder.predict(X_test)
        
    return(X_train,X_test)    #dimension reduced
        
        
def saveLossProgress(history,seed):

    loss_collector, loss_max_atTheEnd = saveLossProgress_ylim(history)

    # save loss progress - train and val loss only
    figureName = 'data_' + str(seed)
    plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'],
               loc='upper right')
    plt.savefig("results/" + figureName + '.png')
    plt.close()

    if 'recon_loss' in history.history:
        figureName = 'data_' + str(seed) + '_detailed'
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
        plt.savefig("results/" + figureName + '.png')
        plt.close()

# supporting loss plot
def saveLossProgress_ylim(history):
    loss_collector = []
    loss_max_atTheEnd = 0.0
    for hist in history.history:
        current = history.history[hist]
        loss_collector += current
        if current[-1] >= loss_max_atTheEnd:
            loss_max_atTheEnd = current[-1]
    return loss_collector, loss_max_atTheEnd        
    
    
X_train_d,X_test_d=pca(X_train, X_test,ncomp=30)
#X_train_d,X_test_d=ae(X_train, X_test,y_train,seed=1,dims=[200,100,10],patience=100)

'''
'''
res_dic={}
for i in np.arange(0.3,1,0.1):
    X_train_d,X_test_d=pca(X_train, X_test,ratio=i)

    clf = RandomForestClassifier(max_depth=None, random_state=0,class_weight="balanced")

    param_grid = {
        'n_estimators': [s for s in range(1, 500, 10)],
    }

    scoring={"balanced Acc":make_scorer(balanced_accuracy_score)}

    search = GridSearchCV(clf, param_grid, cv=StratifiedKFold(10), scoring=scoring, n_jobs=-1, verbose=1,refit="balanced Acc")

    search.fit(X_train_d, y_train.ravel())
    
    results = search.cv_results_
    df=pd.DataFrame(results)   
    mx_acc=df.loc[:,"mean_test_balanced Acc"].max()
    res_dic[i]=mx_acc
'''
#clf.fit(X_train_d, y_train.ravel())
#y_pred=clf.predict(X_test_d)
#print(confusion_matrix(y_test,y_pred))
#print("Balanced Accuracy", balanced_accuracy_score(y_test,y_pred))

'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train_d, y_train)
print(reg.score(X_test_d,y_test))
'''

'''
from sklearn.svm import SVR
res_dic={}
for i in np.arange(0.5,1,0.1):
    X_train_d,X_test_d=pca(X_train, X_test,ratio=i)
    
    hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 1)], 'kernel': ['linear'],'epsilon': [s for s in np.arange(0.01,0.3,0.02)]},
                            {'C': [2 ** s for s in np.arange(-7, 3, 0.5)], 'gamma': [2 ** s for s in np.arange(3, -15, -0.5)],'kernel': ['rbf'],'epsilon': [s for s in np.arange(0.01,0.3,0.02)]}]
                            

    clf = GridSearchCV(SVR(), hyper_parameters, cv=10, scoring='r2', n_jobs=-1, verbose=1,refit=True,return_train_score=True)
    clf.fit(X_train_d, y_train.ravel())
    results = clf.cv_results_
    df=pd.DataFrame(results)   
    mx_acc=df.loc[:,"mean_test_score"].max()
    res_dic[i]=mx_acc
    
    
print(clf.score(X_test_d,y_test))
'''
