#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
data<-read.csv(args[1],row.names = 1,sep='\t')

# data<-read.csv("./../Results/Datasets/III.csv",row.names = 1,sep='\t')

#Modifying dtype
class=colnames(data)[1]
data[[class]]<-factor(data[[class]])

passed=c()
for(i in colnames(data)[-1]){
t<-wilcox.test(data[[i]]~data[[class]],exact=TRUE)
if(t$p.value<=args[3]){passed=c(passed,i)}
}
print("Selected Features")
print(passed)
f_sel_data<-data[,passed]
write.csv(f_sel_data,args[2])