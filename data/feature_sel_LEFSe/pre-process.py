import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df=pandas.read_csv("./../../../Data_21Dec20/species_data.csv",index_col=0)

y1=pandas.read_csv("./../../../METADATA/data_194.csv",index_col=0)
y2=pandas.read_csv("./../../../METADATA/data_test.csv",index_col=0)


#data_normalisation - rel abund
df_norm=(df.div(df.sum(axis=1),axis=0))*100

#Select extremeties 

##Merge Train and test

y=pandas.concat([y1,y2],axis=0)


y=y[(y.ExacerbatorState=="FreqEx") | (y.ExacerbatorState=="NonEx")]
y=y.replace({"NonEx":0,"FreqEx":1})
train=df_norm.reindex(y.index)

train=pandas.merge(train,y["ExacerbatorState"],left_index=True,right_index=True)


train.to_csv("./to_lefse.csv",sep='\t')
#Find and replace " " by "_" in excel
