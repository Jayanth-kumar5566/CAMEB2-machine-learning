import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle 

df=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/MASTER-TABLES/VIRFINDER/c10k_abundance_individual_contigs.csv",index_col=0)



df.drop(["Blank","13LTBlank","76LTBlank"],axis=1,inplace=True)
#df.drop(["Unassigned"],axis=0,inplace=True)

df=df.transpose()


y1=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv",index_col=0)
y2=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv",index_col=0)


#data_normalisation - rel abund
df_norm=(df.div(df.sum(axis=1),axis=0))*100
df_norm=df_norm.fillna(0)

#Filtering
ind=(df_norm>=0)
f_ind=(ind.sum(axis=0)>=13) #5% of the population 
df_norm=df_norm.loc[:,f_ind]

df_norm=(df_norm.div(df_norm.sum(axis=1),axis=0))*100
df_norm=df_norm.fillna(0)
##Merge Train and test

y=pandas.concat([y1,y2],axis=0)


#y=y[(y.ExacerbatorState=="FreqEx") | (y.ExacerbatorState=="NonEx")]
y=y.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
train=df_norm.reindex(y.index)

train=pandas.merge(y["ExacerbatorState"],train,left_index=True,right_index=True)
'''
#Create a dictionary
dic_=dict()
for i in range(len(train.columns)):
	dic_["S"+str(i)]=train.columns[i]

train.columns=dic_.keys()

file_pi = open('/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V_index_dict_.obj', 'wb') 
pickle.dump(dic_, file_pi)
file_pi.close()
'''
train.to_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.csv",sep='\t')
