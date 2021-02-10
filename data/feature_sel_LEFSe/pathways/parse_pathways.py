#!/usr/bin/env python3

import sys
import pandas
import pickle 

filehandler = open("index_dict_.obj", 'rb') 
dict_ = pickle.load(filehandler)

args=sys.argv

file=open(args[1],'r')

out=open(args[2],'w')
out.write('Features\tvalue\tgroup\n')
lines=file.readlines()
for i in lines:
	y=i.split('\t')[3]
	if y!= "":
		x=(i.split('\t')[0])
		x=dict_[str(x)]
		g=i.split('\t')[2]
		out.write(str(x)+"\t"+str(y)+"\t"+str(g)+'\n')

out.close()
