#!/usr/bin/env python3

'''
Run this code as 

./data_exp.py input_dataset output_dir

input_dataset -> pre-processed dataset
output_dir -> output directory

'''

import pandas
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.stats.distance import permanova
from sklearn.metrics.pairwise import euclidean_distances
from skbio import DistanceMatrix
from sklearn.decomposition import PCA

args=sys.argv

data=pandas.read_csv(args[1],index_col=0,sep="\t")
groups=data.loc[:,"ExacerbatorState"]
data=data.loc[:,data.columns != "ExacerbatorState"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_std=False) #This is needed because each value is deviation from mean and larger deviation stronger the signal.
scaler.fit(data)
data=scaler.transform(data)

dist=euclidean_distances(data,data)
dist=DistanceMatrix(dist.round(3))
res=permanova(dist,groups)
print(res)

#Plotting
pca=PCA(n_components=2)
pca.fit(data)
print("Explained variance of PCA")
exv=pca.explained_variance_ratio_*100
print(exv)

data_d=pca.transform(data)


plt.figure(figsize=[8,8])
colors = ['navy', 'darkorange']
target_names=["Non-Frequent exacerbators","Frequent exacerbators"]
lw = 2

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(data_d[groups == i, 0], data_d[groups == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of CAMEB2')
plt.xlabel("PC1 (explained variance = "+str(round(exv[0],3))+"%)")
plt.ylabel("PC2 (explained variance = "+str(round(exv[1],3))+"%)")
plt.figtext(0.7,0.7,str(res),fontsize=5)
plt.tight_layout()
plt.savefig(args[2]+"pca.png",dpi=600)
