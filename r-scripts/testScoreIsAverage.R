#!/usr/bin/env Rscript

library(dplyr)
library(magrittr)
library(ggplot2)

real = c(1,1,0,1)

binarylogloss = function(real, predicted){
    inside = real*log(predicted) + (1-real)*log(1-predicted)
    -1/length(real) * sum(inside)
}

setwd("~/downloads/lungCancer/KAGGLE-lung17")

trainDF = read.csv("stage1_labels.csv")
testDF = read.csv("stage1_sample_submission.csv")


probabilityOfCancer = sum(trainDF$cancer) / nrow(trainDF)

testDF$cancer = probabilityOfCancer

write.csv(testDF, "naive", quote=F, row.names=F)
