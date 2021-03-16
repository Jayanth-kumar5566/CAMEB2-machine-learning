#Reading the dataset
# data=read.csv("./Microbes.csv",row.names = 1)
library(foreach)
library(doParallel)
library(abind)

spearman<-function(data){
  data<-t(data)
  # print("started")

    #Bootstrap
  max_iter=100
  rows=rownames(data)

#  boot_arr<-array(dim=c(dim(data)[2],dim(data)[2],max_iter))
#  for (i in 1:max_iter){
#    t_x=sample(rows,replace = TRUE)
#    sim=cor(data[t_x,],method="spearman")
#    boot_arr[,,i]<-sim
#  }
  
    acomb <- function(...) abind(..., along=3)
    cores=detectCores()
    cl <- makeCluster(cores[1])
    registerDoParallel(cl)
    
  boot_arr<-foreach(i=1:max_iter,.combine=acomb, .multicombine=TRUE) %dopar% {
    t_x=sample(rows,replace = TRUE)
    sim=cor(data[t_x,],method="spearman")
    sim
    }

  # mean_sim<-apply(boot_arr, c(1,2), mean,na.rm=TRUE) #mean of bootstrap values
  
  mean_sim<-parApply(cl,boot_arr, c(1,2), mean,na.rm=TRUE)
  
  #perm and renorm
  # max_iter=100
  # perm_arr<-array(dim=c(dim(data)[2],dim(data)[2],max_iter))
  # for (i in 1:max_iter){
  #   x<-apply(data,2,FUN =sample) #permutation
  #   x<-x/rowSums(x)#renormalization
  #   sim=cor(x,method="spearman")
  #   perm_arr[,,i]<-sim
  # }
  
  perm_arr<-foreach(i=1:max_iter,.combine=acomb, .multicombine=TRUE) %dopar% {
    x<-apply(data,2,FUN =sample) #permutation
    x<-x/rowSums(x)#renormalization
    sim=cor(x,method="spearman")
    sim
  }
  
  # p_val=array(dim=c(dim(data)[2],dim(data)[2]))
  # for (i in 1:dim(data)[2]){
  #   for (j in 1:dim(data)[2]){
  #     t=wilcox.test(perm_arr[i,j,],boot_arr[i,j,],alternative = "two.sided",paired = FALSE,exact = TRUE)
  #     p_val[i,j]<-t$p.value
  #   }
  # }
  
  p_val <- foreach(i=1:dim(data)[2], .combine='rbind') %:%
    foreach(j=1:dim(data)[2], .combine='c') %dopar% {
      t <- wilcox.test(perm_arr[i,j,],boot_arr[i,j,],alternative = "two.sided",paired = FALSE,exact = TRUE)
      p<-t$p.value
      p
    }

  stopCluster(cl)  
  
  p<-p.adjust(p_val,method = "fdr")
  dim(p)<-dim(p_val)
  
  ind<-(p>0.001) #p-value is non-significant
  mean_sim[ind]<-0 #make those edges 0
  return(mean_sim)
}


