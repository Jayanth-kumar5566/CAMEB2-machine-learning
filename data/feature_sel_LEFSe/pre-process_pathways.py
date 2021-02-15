import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle 

df=pandas.read_csv("./../../../MASTER-TABLES/HUMANN2/humann2_unipathway_pathabundance_relab.tsv",sep='\t',index_col=0)
pathways=[i.find("|")==-1 for i in df.index]
df=df.loc[pathways,:]

df.columns=[i.split("_")[0] for i in df.columns]
df.drop(["13LTBlank","76LTBlank","Blank"],axis=1,inplace=True)
df.drop(["UNMAPPED","UNINTEGRATED"],axis=0,inplace=True)

df=(df.div(df.sum(axis=0),axis=1))*100

df["Super_pathway"]=[i.split(";")[0] for i in df.index]
df=df.groupby("Super_pathway")
df=df.sum()

df=df.transpose()


y1=pandas.read_csv("./../../../METADATA/data_194.csv",index_col=0)
y2=pandas.read_csv("./../../../METADATA/data_test.csv",index_col=0)


#data_normalisation - rel abund
df_norm=(df.div(df.sum(axis=1),axis=0))*100

#Filtering
ind=(df_norm>0)
f_ind=(ind.sum(axis=0)>=13) #5% of the population 
df_norm=df_norm.loc[:,f_ind]

df_norm=(df_norm.div(df_norm.sum(axis=1),axis=0))*100

##Merge Train and test

y=pandas.concat([y1,y2],axis=0)


#y=y[(y.ExacerbatorState=="FreqEx") | (y.ExacerbatorState=="NonEx")]
y=y.replace({"NonEx":0,"Exacerbator":1,"FreqEx":1})
train=df_norm.reindex(y.index)

train=pandas.merge(train,y["ExacerbatorState"],left_index=True,right_index=True)

#Create a dictionary
dic_=dict()
for i in range(len(train.columns)):
	dic_["S"+str(i)]=train.columns[i]

train.columns=dic_.keys()

file_pi = open('./pathways/index_dict_.obj', 'wb') 
pickle.dump(dic_, file_pi)
file_pi.close()

train.to_csv("./pathways/to_lefse_pathways.csv",sep='\t')
