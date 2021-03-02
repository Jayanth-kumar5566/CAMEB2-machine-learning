#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
print(args)

library(ggpubr)
library(ggplot2)
data<-read.csv(args[1])
data$group<-factor(data$group)

png(args[2],width = 12,height = 12,res = 300,units="in")
ggbarplot(data, x = "Features", y = "value",
          fill = "group",           # change fill color by mpg_level
          color = "black",            # Set bar border colors to white
          palette = "jco",            # jco journal color palett. see ?ggpar
          sort.val = "asc",          # Sort the value in descending order
          sort.by.groups = TRUE,     # Don't sort inside each group
          x.text.angle = 0,          # Rotate vertically x axis texts
          ylab = "LDA Score",
          legend.title = "Clusters",
          rotate = TRUE
)+scale_y_continuous(expand = expansion(mult = c(0, .1)))
dev.off()
