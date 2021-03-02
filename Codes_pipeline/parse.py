#!/usr/bin/env python3

import sys
import pandas
args=sys.argv
df=pandas.read_csv(args[1],sep='\t',index_col=0)

file=open(args[2],'r')

out=open(args[3],'w')
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
