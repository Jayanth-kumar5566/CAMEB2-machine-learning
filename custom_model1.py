#Implementing random forests with feature engineering

# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skbio.stats.composition import clr
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


#args
args=sys.argv

#Reading the data - training
df=pd.read_csv("./../Data_21Dec20/species_data.csv",index_col=0)
y_train=pd.read_csv("./../METADATA/data_194.csv",index_col=0)
y_test=pd.read_csv("./../METADATA/data_test.csv",index_col=0)

# Feature-selection LEFSe
f=pd.read_csv("./data/feature_sel_LEFSe/selected_microbes.csv",index_col=0).index
df_sel=df.loc[:,f]
#df_sel=df.loc[:,:]
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
f=pd.read_csv("./data/feature_sel_LEFSe/pathways/selected_pathways.csv",index_col=0).index
df_sel=df.loc[df_norm.index,f]
#df_sel=df.loc[df_norm.index,:]
del df,f

#Pathway_norm
pdf_norm=clr(df_sel+1)
pdf_norm=pd.DataFrame(pdf_norm,index=df_sel.index,columns=df_sel.columns)
del df_sel

#AMR

df=pd.read_csv("./../MASTER-TABLES/AMR/shortbred-CARD-95-summary.csv",index_col=0)
df=df.groupby("Drug.ClassAMR")
df=df.sum()

df.drop(["Blank"],axis=1,inplace=True)

df=df.transpose()

# Feature-selection LEFSe
f=pd.read_csv("./data/feature_sel_LEFSe/amr/selected_amr.csv",index_col=0).index
df_sel=df.loc[df_norm.index,f]
#df_sel=df.loc[df_norm.index,:]
del df,f

#AMR_norm
adf_norm=clr(df_sel+1)
adf_norm=pd.DataFrame(adf_norm,index=df_sel.index,columns=df_sel.columns)
del df_sel

#Phage
df=pd.read_csv("./../MASTER-TABLES/VIRFINDER/c10k_abundance_demovir_Family.csv",index_col=0)

df.drop(["Blank","X13LTBlank","X76LTBlank"],axis=1,inplace=True)
df.drop(["Unassigned"],axis=0,inplace=True)

df=df.transpose()

# Feature-selection LEFSe
f=pd.read_csv("./data/feature_sel_LEFSe/phage/selected_phage.csv",index_col=0).index
df_sel=df.loc[df_norm.index,f]
#df_sel=df.loc[df_norm.index,:]
del df,f

#phage_norm
vdf_norm=clr(df_sel+1)
vdf_norm=pd.DataFrame(vdf_norm,index=df_sel.index,columns=df_sel.columns)
del df_sel

#Merge dataframes
data=pd.merge(df_norm,pdf_norm,left_index=True,right_index=True)
data=pd.merge(data,adf_norm,left_index=True,right_index=True)
data=pd.merge(data,vdf_norm,left_index=True,right_index=True)

#Training and testing splitting
X_train_d=data.reindex(y_train.index)
X_test_d=data.reindex(y_test.index)
del data
#y_test and train - class replacment
y_train=y_train["ExacerbatorState"]
y_train=y_train.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
y_test=y_test["ExacerbatorState"]
y_test=y_test.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})


#Principal Component Analysis
def pca(X_train,X_test,ratio=0.99):
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


def saveLossProgress(history):
        loss_collector, loss_max_atTheEnd = saveLossProgress_ylim(history)
        figureName = "vae_plot.png"
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
            plt.savefig(figureName + '.png')
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
        return(X_train_m,X_test_m,history)

#Dimensionality reduction
'''
D_X_train_d,D_X_test_d,history= vae(X_train_d,y_train,X_test_d,dims=[80,20])
saveLossProgress(history)
'''
D_X_train_d,D_X_test_d=X_train_d,X_test_d

#D_X_train_d,D_X_test_d = pca(X_train_d,X_test_d,ratio=0.99)

#Logistic-regression
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

'''
# define models and parameters
solvers = ['liblinear']
penalty = ['l1']
#c_values = [100, 10, 1.0, 0.1, 0.01,0.001]
c_values = np.linspace(0.01,10,50)

#scoring={"Acc":make_scorer(accuracy_score)}
scoring={"Acc":make_scorer(balanced_accuracy_score)}
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
model = LogisticRegression(fit_intercept=True,class_weight="balanced")
if args[1]=="LOOCV":
    clf = GridSearchCV(estimator=model, param_grid=grid, cv=LeaveOneOut(), scoring=scoring,n_jobs=-1,return_train_score=True,refit=False)
elif args[1]=="RepeatedKfold":
    clf = GridSearchCV(estimator=model, param_grid=grid, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), scoring=scoring,n_jobs=-1,return_train_score=True,refit=False)
    
clf.fit(D_X_train_d, y_train)

'''
#Random Forest
hyper_parameters = [{'n_estimators': [10,150],'criterion':['gini'],
                        'max_features': ['auto'],
                        'max_depth':[None],
                        'min_samples_split':[i for i in np.linspace(0.001,0.5,50)],
                        'min_samples_leaf':[i for i in np.linspace(0.001,0.5,50)],
                        }, ]
scoring={"Acc":make_scorer(balanced_accuracy_score)}

if args[1]=="LOOCV":
    clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced",bootstrap=True), hyper_parameters, cv=LeaveOneOut(), scoring=scoring, n_jobs=-1, verbose=1,refit="Acc",return_train_score=True)
elif args[1]=="RepeatedKfold":
    clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced",bootstrap=True), hyper_parameters, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), scoring=scoring, n_jobs=-1, verbose=1,refit=False,return_train_score=True)

clf.fit(X_train_d, y_train)                         

results = clf.cv_results_
df=pd.DataFrame(results)

if args[1]=="LOOCV":


    df_test=df.filter(regex=("split.*_test_Acc"))


    Acuracy_cbalanced=df_test.apply(score,axis=1)
    Acuracy_cbalanced=pd.DataFrame(Acuracy_cbalanced,columns=["Class_average Accuracy"])
    del df_test


    #===================Training_Acc============================
    train_acc_balanced=df["mean_train_Acc"]

    plt.plot(c_values,train_acc_balanced,label="train")
    plt.plot(c_values,Acuracy_cbalanced,label="test_balanced")
    plt.legend()
    plt.title("l1 norm")
    plt.savefig("lr1_plot.png",dpi=600)


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

elif args[1]=="RepeatedKfold":
    c_values=hyper_parameters[0]['min_samples_split']
    plt.plot(c_values,df["mean_train_Acc"],label="train")
    plt.plot(c_values,df["mean_test_Acc"],label="test_balanced")
    plt.legend()
    plt.title("l1 norm")
    plt.savefig("lr1_plot.png",dpi=600)


plt.plot(df["mean_train_Acc"],df["mean_test_Acc"],'o',markersize=1)
plt.xlabel("Mean balanced train Acc")
plt.ylabel("Mean balanced testn Acc")
plt.savefig("plot_train_test.png",dpi=600)

df_sel=df.loc[:,["params","mean_train_Acc","mean_test_Acc"]]
df_sel.to_csv("params_acc.csv")

'''

lr = LogisticRegression(penalty="l1",class_weight="balanced",solver="liblinear",C=1,n_jobs=-1)
lr.fit(D_X_train_d,y_train)
print("Training Acc",lr.score(D_X_train_d,y_train))
test_score=lr.score(D_X_test_d,y_test)
print("Testing Acc",test_score)
y_pred=lr.predict(D_X_test_d)
print("Confusion Matrix ",confusion_matrix(y_test,y_pred))
print("F Score in weighted fashion ",f1_score(y_test,y_pred,average="weighted"))


'''
#Input the best parameters and then run
x=[]
imp=0
for i in range(100):
    rf=RandomForestClassifier(n_jobs=-1, n_estimators=150,min_samples_split=0.001,max_depth=None,min_samples_leaf
=0.31669,class_weight="balanced",bootstrap=True)
    rf.fit(X_train_d, y_train)
    imp=imp+rf.feature_importances_
    y_pred=rf.predict(X_test_d)
    print(confusion_matrix(y_test,y_pred))
    x.append(rf.score(X_test_d,y_test))

print("Median of testing acc",np.median(x))
f_imp=pd.DataFrame([i for i in zip(X_train_d,imp)],columns=["Features","Importance"])
f_imp.to_csv("feature_importances.csv")

