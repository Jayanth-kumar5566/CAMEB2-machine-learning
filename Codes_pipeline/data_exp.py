#!/usr/bin/env python3

'''
Run this code as 

./data_exp.py input_dataset training_group testing_group

input_dataset -> microbiome datasets

training_group -> dataset with training indices

testing_group -> dataset with testing indices
'''

import pandas
import sys
import matplotlib.pyplot as plt
import seaborn as sns

args=sys.argv


data=pd.read_csv(args[1],index_col=0)
y_train=pd.read_csv(args[2],index_col=0)
y_test=pd.read_csv(args[3],index_col=0)

y=pandas.concat([y_train,y_test],axis=0)
y=y.replace({"NonEx":0,"Exacerbator":0,"FreqEx":1})
data=data.reindex(y.index)
