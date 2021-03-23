library(glmnet)
library(ROCR)
library(MASS)
library(ggplot2)
library(pheatmap)
library(randomForest)


setwd("~/Desktop")
dat <- read.csv("CancerExample.csv", header = TRUE)

##glmnet

#Get Training Set
dat.train <- dat[which(dat$Set == "Training"),]

dat.train.x <- dat.train[,6:ncol(dat.train)]
dat.train.y <- dat.train$Censor

dat.train.y <- as.factor(as.character(dat.train.y))



#Before moving forward with modeling the training data set. Lets use PCA and
#hierarchical clustering to visualize the data and explore 
#


pc.result<-prcomp(dat.train.x,scale.=TRUE)
pc.scores<-pc.result$x
pc.scores<-data.frame(pc.scores)
pc.scores$Censor<-dat.train.y


#Loadings for interpretation
pc.result$rotation

#Scree plot
pc.eigen<-(pc.result$sdev)^2
pc.prop<-pc.eigen/sum(pc.eigen)
pc.cumprop<-cumsum(pc.prop)
plot(1:9,pc.prop,type="l",main="Scree Plot",ylim=c(0,1),xlab="PC #",ylab="Proportion of Variation")
lines(1:9,pc.cumprop,lty=3)

#Use ggplot2 to plot the first few pc's
ggplot(data = pc.scores, aes(x = PC1, y = PC2)) +
  geom_point(aes(col=Censor), size=1)+
  geom_hline(yintercept = 0, colour = "gray65") +
  geom_vline(xintercept = 0, colour = "gray65") +
  ggtitle("PCA plot of Gene Expression Data")

ggplot(data = pc.scores, aes(x = PC1, y = PC3)) +
    geom_point(aes(col=Censor), size=1)+
  geom_hline(yintercept = 0, colour = "gray65") +
    geom_vline(xintercept = 0, colour = "gray65") +
    ggtitle("PCA plot of Gene Expression Data")

  
ggplot(data = pc.scores, aes(x = PC2, y = PC3)) +
    geom_point(aes(col=Censor), size=1)+
  geom_hline(yintercept = 0, colour = "gray65") +
    geom_vline(xintercept = 0, colour = "gray65") +
    ggtitle("PCA plot of Gene Expression Data")
  
#Lets look at a heatmap using hierarchical clustering to see if the 
#response naturually clusters out using the predictors
  
#Transposting the predictor matrix and giving the response categories its
#row names.

library(RColorBrewer)
cols <- colorRampPalette(brewer.pal(9, "Set1"))
x<-t(dat.train.x)
colnames(x)<-dat.train.y
pheatmap(x,annotation_col=data.frame(Cancer=dat.train.y),annotation_colors=list(Cancer=c("0"="white","1"="green")),scale="row",legend=T,color=colorRampPalette(c("blue","white", "red"), space = "rgb")(100))


#A good exercise would be to go in and play around with the 
#data but maybe add a few "fake" predictors that seperate out
#the two response categories very well.  From there make the plots 
#again and see how they reflect that truth.

#The heatmap could be done using the PC's as well.



 
  
  
#glmnet requires a matrix 
dat.train.x <- as.matrix(dat.train.x)
library(glmnet)
cvfit <- cv.glmnet(dat.train.x, dat.train.y, family = "binomial", type.measure = "class", nlambda = 1000)
plot(cvfit)
coef(cvfit, s = "lambda.min")

#Get training set predictions...We know they are biased but lets create ROC's.
#These are predicted probabilities from logistic model  exp(b)/(1+exp(b))
fit.pred <- predict(cvfit, newx = dat.train.x, type = "response")

#Compare the prediction to the real outcome
head(fit.pred)
head(dat.train.y)

#Create ROC curves
pred <- prediction(fit.pred[,1], dat.train.y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values

#Plot ROC
plot(roc.perf)
abline(a=0, b= 1) #Ref line indicating poor performance
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Get Validation Set I
dat.val1 <- dat[which(dat$Set == "Validation I"),]
dat.val1.x <- dat.val1[,c(6:ncol(dat))]
dat.val1.x <- as.matrix(dat.val1.x)

dat.val1.y <- dat.val1$Censor
dat.val1.y <- as.factor(as.character(dat.val1.y))


#Run model from training set on valid set I
fit.pred1 <- predict(cvfit, newx = dat.val1.x, type = "response")

#ROC curves
pred1 <- prediction(fit.pred1[,1], dat.val1.y)
roc.perf1 = performance(pred1, measure = "tpr", x.measure = "fpr")
auc.val1 <- performance(pred1, measure = "auc")
auc.val1 <- auc.val1@y.values
plot(roc.perf1)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.val1[[1]],3), sep = ""))


#Rinse and Repeat for valid set II and III
dat.val2 <- dat[which(dat$Set == "Validation II"),]
dat.val2.x <- dat.val2[,c(6:ncol(dat))]
dat.val2.x <- as.matrix(dat.val2.x)

dat.val2.y <- dat.val2$Censor
dat.val2.y <- as.factor(as.character(dat.val2.y))

fit.pred2 <- predict(cvfit, newx = dat.val2.x, type = "response")

pred2 <- prediction(fit.pred2[,1], dat.val2.y)
roc.perf2 = performance(pred2, measure = "tpr", x.measure = "fpr")
auc.val2 <- performance(pred2, measure = "auc")
auc.val2 <- auc.val2@y.values
plot(roc.perf2)
abline(a=0, b= 1)
text(x = .42, y = .6,paste("AUC = ", round(auc.val2[[1]],3), sep = ""))

#Valid set III
dat.val3 <- dat[which(dat$Set == "Validation III"),]
dat.val3.x <- dat.val3[,c(6:ncol(dat))]
dat.val3.x <- as.matrix(dat.val3.x)

dat.val3.y <- dat.val3$Censor
dat.val3.y <- as.factor(as.character(dat.val3.y))


fit.pred3 <- predict(cvfit, newx = dat.val3.x, type = "response")

pred3 <- prediction(fit.pred3[,1], dat.val3.y)
roc.perf3 = performance(pred3, measure = "tpr", x.measure = "fpr")
auc.val3 <- performance(pred3, measure = "auc")
auc.val3 <- auc.val3@y.values
plot(roc.perf3)
abline(a=0, b= 1)
text(x = .4, y = .6,paste("AUC = ", round(auc.val3[[1]],3), sep = ""))


  
#Here we are just comparing the reproducibility of the logistic model.
#You can also use this code to compare multiple models like logistic, compared to lda, compared to a tree or RF.
#This graph also allows for you to get a sense of what cut off values are producing the best sensitivity and specificity results as well using
#the colorize option.

#This is helpful:  https://www.r-bloggers.com/a-small-introduction-to-the-rocr-package/ that does some extra things you might find 
#helpful.
#If you want to mess around with other packages: https://rviews.rstudio.com/2019/03/01/some-r-packages-for-roc-curves/
plot( roc.perf1, colorize = TRUE)
plot(roc.perf2, add = TRUE, colorize = TRUE)
plot(roc.perf3, add = TRUE, colorize = TRUE)
abline(a=0, b= 1)

#without color for cutoff; but adding colors to allow for comarisons of the curves
plot( roc.perf1)
plot(roc.perf2,col="orange", add = TRUE)
plot(roc.perf3,col="blue", add = TRUE)
legend("bottomright",legend=c("Valid 1","Valid 2","Valid 3"),col=c("black","orange","blue"),lty=1,lwd=1)
abline(a=0, b= 1)




## LDA

#Training Set
dat.train <- dat[which(dat$Set == "Training"),]
dat.train.x <- dat.train[,6:ncol(dat)]

dat.train.y <- dat.train$Censor
dat.train.y <- as.factor(as.character(dat.train.y))

fit.lda <- lda(dat.train.y ~ ., data = dat.train.x)
pred.lda <- predict(fit.lda, newdata = dat.train.x)

preds <- pred.lda$posterior
preds <- as.data.frame(preds)

pred <- prediction(preds[,2],dat.train.y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Valid set I
dat.val1 <- dat[which(dat$Set == "Validation I"),]
dat.val1.x <- dat.val1[,c(6:ncol(dat))]

dat.val1.y <- dat.val1$Censor
dat.val1.y <- as.factor(as.character(dat.val1.y))


pred.lda1 <- predict(fit.lda, newdata = dat.val1.x)

preds1 <- pred.lda1$posterior
preds1 <- as.data.frame(preds1)

pred1 <- prediction(preds1[,2],dat.val1.y)
roc.perf = performance(pred1, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred1, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Valid set II
dat.val2 <- dat[which(dat$Set == "Validation II"),]
dat.val2.x <- dat.val2[,c(5:ncol(dat))]

dat.val2.y <- dat.val2$Censor
dat.val2.y <- as.factor(as.character(dat.val2.y))

pred.lda2 <- predict(fit.lda, newdata = dat.val2.x)

preds2 <- pred.lda2$posterior
preds2 <- as.data.frame(preds2)

pred2 <- prediction(preds2[,2],dat.val2.y)
roc.perf = performance(pred2, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred2, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Valid set III
dat.val3 <- dat[which(dat$Set == "Validation III"),]
dat.val3.x <- dat.val3[,c(5:ncol(dat))]

dat.val3.y <- dat.val3$Censor
dat.val3.y <- as.factor(as.character(dat.val3.y))


pred.lda3 <- predict(fit.lda, newdata = dat.val3.x)

preds3 <- pred.lda3$posterior
preds3 <- as.data.frame(preds3)

pred3 <- prediction(preds3[,2],dat.val3.y)
roc.perf = performance(pred3, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred3, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))



#One more time with a Random Forest
#I'm removing the fluff variables just to make coding easier.
dat.train.rf <- dat[which(dat$Set == "Training"),-(1:4)]
dat.train.rf$Censor<-factor(dat.train.rf$Censor)

train.rf<-randomForest(Censor~.,data=dat.train.rf,mtry=4,ntree=500,importance=T)
fit.pred<-predict(train.rf,newdata=dat.train.rf,type="prob")



pred <- prediction(fit.pred[,2], dat.train.rf$Censor)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Predict Validation Set I
dat.val1.rf <- dat.val1[,-(1:4)]
#dat.val1.rf$Censor<-factor(dat.train.rf$Censor)

pred.val1<-predict(train.rf,newdata=dat.val1.rf,type="prob")



pred <- prediction(pred.val1[,2], dat.val1.rf$Censor)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Predict Validation Set 2
dat.val2.rf <- dat.val2[,-(1:4)]
#dat.val1.rf$Censor<-factor(dat.train.rf$Censor)

pred.val2<-predict(train.rf,newdata=dat.val2.rf,type="prob")



pred <- prediction(pred.val2[,2], dat.val2.rf$Censor)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Predict Validation Set 3
dat.val3.rf <- dat.val3[,-(1:4)]
#dat.val1.rf$Censor<-factor(dat.train.rf$Censor)

pred.val3<-predict(train.rf,newdata=dat.val3.rf,type="prob")



pred <- prediction(pred.val3[,2], dat.val3.rf$Censor)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))









#Suppose we did not have all of these validation data sets.  We can assess how well our model building process works through Cross validation.
#The idea is that we can get an idea of how well the approach is going to perform on new data not yet collected.
#We will use AUC as the performance matrix.  Note we only consider the lasso logistic here, but in practice we could 
#run many different models to compare directly.

nloops<-50   #number of CV loops
ntrains<-dim(dat.train.x)[1]  #No. of samples in training data set
cv.aucs<-c() #initializing a vector to store the auc results for each CV run

for (i in 1:nloops){
 index<-sample(1:ntrains,60)
 cvtrain.x<-as.matrix(dat.train.x[index,])
 cvtest.x<-as.matrix(dat.train.x[-index,])
 cvtrain.y<-dat.train.y[index]
 cvtest.y<-dat.train.y[-index]
 
 cvfit <- cv.glmnet(cvtrain.x, cvtrain.y, family = "binomial", type.measure = "class") 
 fit.pred <- predict(cvfit, newx = cvtest.x, type = "response")
 pred <- prediction(fit.pred[,1], cvtest.y)
 roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
 auc.train <- performance(pred, measure = "auc")
 auc.train <- auc.train@y.values
 
 cv.aucs[i]<-auc.train[[1]]
}

hist(cv.aucs)
summary(cv.aucs)




#Doing the same procedure for random allocation of response values.
#Good practice when number of yes/no is not balanced.


nloops<-50   #number of CV loops
ntrains<-dim(dat.train.x)[1]  #No. of samples in training data set
cv.aucs<-c()
dat.train.yshuf<-dat.train.y[sample(1:length(dat.train.y))]

for (i in 1:nloops){
  index<-sample(1:ntrains,60)
  cvtrain.x<-as.matrix(dat.train.x[index,])
  cvtest.x<-as.matrix(dat.train.x[-index,])
  cvtrain.y<-dat.train.yshuf[index]
  cvtest.y<-dat.train.yshuf[-index]
  
  cvfit <- cv.glmnet(cvtrain.x, cvtrain.y, family = "binomial", type.measure = "class") 
  fit.pred <- predict(cvfit, newx = cvtest.x, type = "response")
  pred <- prediction(fit.pred[,1], cvtest.y)
  roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
  auc.train <- performance(pred, measure = "auc")
  auc.train <- auc.train@y.values
  
  cv.aucs[i]<-auc.train[[1]]
}

hist(cv.aucs)
summary(cv.aucs)
