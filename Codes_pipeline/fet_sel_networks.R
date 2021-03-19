#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
data<-read.csv("./../Results/Datasets/VI.csv",row.names = 1)

#Exacerbation data
y1=read.csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv",row.names=1)
y2=read.csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv",row.names=1)
y=rbind.data.frame(y1,y2)
y$ExacerbatorState<-factor(ifelse(y$Exacerbations<3,0,1))

data<-data[row.names(y),]
data$ExacerbatorState<-y$ExacerbatorState

rm(y,y1,y2)


library(foreach)
library(doParallel)

cores=detectCores()
cl <- makeCluster(cores[1])
registerDoParallel(cl)

#Filetring
ind<-(parApply(cl,data, 2, var,na.rm=TRUE)!=0)
data<-data[,ind]

passed<-foreach(i=colnames(data)[-dim(data)[2]],.combine=c) %dopar% {
  t<-wilcox.test(data[[i]]~data[["ExacerbatorState"]],exact=TRUE)
  if(t$p.value<=args[2]){
    i
  }
}

stopCluster(cl)  

print("Selected Features")
print(passed)
f_sel_data<-data[,passed]
f_sel_data[is.na(f_sel_data)] <- 0
write.csv(f_sel_data,args[1])