import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/Data_21Dec20/species_data.csv",index_col=0)

y1=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv",index_col=0)
y2=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv",index_col=0)


#data_normalisation - rel abund
df_norm=clr(df+1)
df_norm=pandas.DataFrame(df_norm,index=df.index,columns=df.columns)

#df_norm=df
#Select extremeties 

##Merge Train and test

y=pandas.concat([y1,y2],axis=0)

#y=y[(y.ExacerbatorState=="FreqEx") | (y.ExacerbatorState=="NonEx")]
y=y.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
train=df_norm.reindex(y.index)

train=pandas.merge(y["ExacerbatorState"],train,left_index=True,right_index=True)
train.columns=[i.replace(" ","_") for i in train.columns]

train.to_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.csv",sep='\t')
