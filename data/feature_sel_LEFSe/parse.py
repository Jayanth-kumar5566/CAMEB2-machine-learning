#!/usr/bin/env python3

import sys
import pandas
df=pandas.read_csv("to_lefse.csv",sep='\t',index_col=0)


args=sys.argv

file=open(args[1],'r')

out=open(args[2],'w')
out.write('Features,value,group\n')
lines=file.readlines()
for i in lines:
	y=i.split('\t')[3]
	if y!= "":
		x=(i.split('\t')[0])
		x=x.replace("_"," ")
		g=i.split('\t')[2]
		out.write(str(x)+","+str(y)+","+str(g)+'\n')

out.close()
