#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
data<-read.csv(args[1],row.names = 1,sep='\t')

# data<-read.csv("./../Results/Datasets/I.csv",row.names = 1,sep='\t')

#Modifying dtype
categ<-c("ExacerbatorState","Sex","ICS.use","Oral.ab","Smoker","Aetiology","BCOS","Pseudomonas.culture.positive","Bronchodialator","Mucolytic","Oral.steroids","Long.term.antibiotics")

for(i in categ){data[[i]]<-factor(data[[i]])}
data[["Other.organisms.isolated.in.sputum"]]<-NULL #Not proper

passed=c()
for(i in colnames(data)[-1]){
  if(is.numeric(data[[i]])){
  t<-wilcox.test(data[[i]]~data$ExacerbatorState)}
else{
  t<-chisq.test(data[[i]],data$ExacerbatorState,simulate.p.value = TRUE)
}
if(t$p.value<=0.05){passed=c(passed,i)}
}
print("Selected Features")
print(passed)
f_sel_data<-data[,passed]
write.csv(f_sel_data,args[2])