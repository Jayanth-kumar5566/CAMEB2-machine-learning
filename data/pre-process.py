import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df=pandas.read_csv("./../../Data_21Dec20/species_data.csv",index_col=0)

y1=pandas.read_csv("./../../METADATA/data_194.csv",index_col=0)
y2=pandas.read_csv("./../../METADATA/data_test.csv",index_col=0)

#data_normalisation
df_norm=clr(df+1)
df_norm=pandas.DataFrame(df_norm,index=df.index,columns=df.columns)

#training
train=df_norm.reindex(y1.index)
train.to_csv("./train.csv")

y1=y1["ExacerbatorState"]
y1=y1.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
y1.to_csv("./train_labels.csv")

#testing
test=df_norm.reindex(y2.index)
test.to_csv("./test.csv")

y2=y2["ExacerbatorState"]
y2=y2.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
y2.to_csv("./test_labels.csv")

