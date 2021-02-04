df=read.csv("./../../../Data_21Dec20/species_data.csv",row.names = 1)
y1=read.csv("./../../../METADATA/data_194.csv",row.names=1)
y2=read.csv("./../../../METADATA/data_test.csv",row.names=1)

library(selbal)
library(compositions)

train_ind<-row.names(y1)
test_ind<-row.names(y2)
ind<-c(train_ind,test_ind)
df<-df[ind,]

labels<-c(as.character(y1$ExacerbatorState),as.character(y2$ExacerbatorState))
labels[labels=="Exacerbator"]="NonEx"
labels<-factor(labels)


selbal(x = df, y = labels,zero.rep="one")

