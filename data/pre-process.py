import pandas
from skbio.stats.composition import clr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df=pandas.read_csv("./../../../Data/Pre-processing/Metaphlan/merged_df.csv",index_col=0)

#data filtering
def ab_pre_filter(df):
    df_norm=(df.div(df.sum(axis=1),axis=0))*100
    ind=((df_norm>0).sum(axis=0))>=10 #5% of 194
    return(df.loc[:,ind])

df_fil=ab_pre_filter(df.iloc[:,1:])

#data_normalisation
df_norm=clr(df_fil+1)
#df_norm=(df_fil.div(df_fil.sum(axis=1),axis=0))*100 #relabund

#scaler=StandardScaler()
#df_norm=scaler.fit_transform(df_norm) #standardization doesn't affect the results

#columns=(["s__"+i for i in df_fil.columns])
df_norm=pandas.DataFrame(df_norm,index=df.index,columns=df_fil.columns)

'''
ind=(df["ExacerbatorState"]=="FreqEx")
ind1=(df["ExacerbatorState"]=="NonEx")
ind2=(df["ExacerbatorState"]=="Exacerbator")

r_c=round(df_fil.shape[1]/2)

fig,ax = plt.subplots(r_c,r_c,figsize=(24,24))
ax=ax.flatten()
for i in range(df_fil.shape[1]):
    ax[i].hist(df_norm[ind1].iloc[:,i],color="blue",density=True,alpha=0.5)
    ax[i].hist(df_norm[ind].iloc[:,i],color="red",density=True,alpha=0.5)
    ax[i].hist(df_norm[ind2].iloc[:,i],color="green",density=True,alpha=0.5)
#    (stat,pvalue)=mannwhitneyu(df_norm[ind1].iloc[:,i],df_norm[ind].iloc[:,i],alternative="two-sided")
    (stat,pvalue)=kruskal(df_norm[ind1].iloc[:,i],df_norm[ind].iloc[:,i],df_norm[ind2].iloc[:,i])
    if pvalue<0.05:
    	print(df_norm.columns[i])
    	ax[i].set_title(df_norm.columns[i],color="red")
    else:
    	ax[i].set_title(df_norm.columns[i])
    ax[i].text(0.75,0.9,"pvalue="+str(round(pvalue,4)),transform=ax[i].transAxes,horizontalalignment='center',verticalalignment='center')
plt.savefig("fig.png",dpi=300)

'''
df_norm.to_csv("./data.csv",header=False,index=False)

df=df["ExacerbatorState"]
df=df.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
df.to_csv("./labels.csv",index=False)
