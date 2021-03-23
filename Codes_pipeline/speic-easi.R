#============Function==============
library(SpiecEasi)
library(Matrix)
library(igraph)
spe<-function(X_fil){
  print("se started")
  X_fil<-t(X_fil)
  names<-colnames(X_fil)
  se <- spiec.easi(as.matrix(X_fil), method='glasso',lambda.min.ratio=1e-3, nlambda=30, pulsar.params=list(rep.num=20,thresh = 0.05,ncores=16))
  secor  <- cov2cor(getOptCov(se))
  adj_matrix <- secor*getRefit(se)
  row.names(adj_matrix)<-names
  colnames(adj_matrix)<-names
  return(as.matrix(adj_matrix))
  # elist.gl[,1]<-names[elist.gl[,1]]
  # elist.gl[,2]<-names[elist.gl[,2]]
  # #edge-list to adjacency matrix
  # G<-graph.data.frame(elist.gl, directed=T, vertices=names)
  # E(G)$weight<-elist.gl[,3]
  # res<-as_adjacency_matrix(G,attr="weight")
  # return(as.matrix(res))
}
