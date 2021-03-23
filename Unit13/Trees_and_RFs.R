library(ggplot2)
library(rgl) ##For mac users you may need to download Xquartz before the 3d plots
#will run.
library(tree)
library(ISLR)
library(randomForest)

#setwd("Z:/STA328/Rlabs")
setwd("~/Desktop/PredMod/Lecture3")


Adver<-read.csv("Advertising.csv",header=T)
Adver<-Adver[,-1]
attach(Adver)
View(Adver)


#Lets make a pretty plot for the predictor space along with the response
#Top down view
library(ggplot2)
p1<-ggplot(Adver,aes(x=radio,y=TV,fill=sales))
p1+geom_point(aes(color=sales))
p1+geom_point(shape=21,size=2.5)+scale_fill_gradient(low='white', high='blue')





#plotting regression fit of sales 
#illustrating predictor space
simple.fit<-lm(sales~TV+radio)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,outer(0:300,0:50,function(x,y) {2.92+.04575*x+.18799*y}),alpha=.4)


#adding interaction to get a better fit
inter.fit<-lm(sales~TV+radio+TV:radio)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,outer(0:300,0:50,function(x,y) {6.75+.019*x+.0288*y+.0011*x*y}),alpha=.4)


#Now lets get to a decision tree fit
#Lets make a simple tree forcing the node sizes to have at least 50 observations
short.tree<-tree(sales~TV+radio,Adver,mincut=50)
summary(short.tree)
plot(short.tree)
text(short.tree)

#View the regions based on the simple tree
p2<-p1+geom_point(shape=21,size=2.5)+scale_fill_gradient(low='white', high='blue')

p2+geom_hline(yintercept=122.05)+geom_segment(x=26.85,y=122.05,xend=26.85,yend=305)


#View the prediction just so we can get a sense of whats going on.
predictors<-data.frame(TV=rep(0:300,51),radio=rep(0:50,each=301))
pred.surface<-matrix(predict(short.tree,newdata=predictors),301,51)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,pred.surface,alpha=.4)



#Lets work under a more normal scenario of fitting a decision tree.
#We will fit pretty deep tree and then perform CV to determ the optimal
#pruned tree.  We could also split into training and test and compare test
#error metrics.


par(mfrow=c(1,1))
mytree<-tree(sales~TV+radio,Adver) #default is for atleast 5 observations in the child nodes for split to occur
summary(mytree)
plot(mytree)
text(mytree,pretty=0)


#Since the tree could likely overfit....
#Lets perform cross validatoin to determine the optimal pruned tree
#Just like LASSO, CV is done by penalizing the Residual Sums of Squares (Deviance)
#based on the total number of nodes present.
#CV plots are typically plotted as a function of the number of nodes 
#the tree has (tree size)

set.seed(3)
cv.adver=cv.tree(mytree,FUN=prune.tree,method="deviance")
names(cv.adver)
cv.adver
plot(cv.adver)
par(mfrow=c(1,1))
plot(cv.adver$size,cv.adver$dev,type="b")
#plot(cv.adver$k,cv.adver$dev,type="b")
#
#CV informs us we should not prune at all in this case.  
#If the minimal test error is on a small size tree,just prune the
#the tree to that size using "best" in the prune.tree function.
prune.adver=prune.tree(mytree,best=8)
plot(prune.adver)
text(prune.adver,pretty=0)

#View the prediction just so we can get a sense of whats going on.
predictors<-data.frame(TV=rep(0:300,51),radio=rep(0:50,each=301))
pred.surface<-matrix(predict(prune.adver,newdata=predictors),301,51)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,pred.surface,alpha=.4)



#Lets do a trian/test set split and compare knn,regular regression,
#and a regression tree's test MSE.
set.seed(123)
index<-sample(1:200,100)
train<-Adver[index,]
test<-Adver[-index,]

train.tree<-tree(sales~TV+radio,train)
summary(train.tree)
plot(train.tree)
text(train.tree,pretty=0)
plot(cv.tree(train.tree,FUN=prune.tree,method="deviance"))

testMSE<-mean((test$sales- predict(train.tree,newdata=test) )^2)


knn<-FNN::knn.reg(train = train[,-c(3,4)], test =test[,-c(3,4)], y = train$sales, k = 5)
testMSE.knn<-mean( ( test$sales-knn$pred)^2)

full.fit<-lm(sales~TV+radio+TV:radio,train)
testMSE.ols<-mean((test$sales-predict(full.fit,test))^2)


testMSE
testMSE.knn
testMSE.ols


#Now lets move on to bagging and random forrest using the same data set and see
#How we can improve over a simple decision tree.


train<-Adver[index,]
test<-Adver[-index,]

par(mfrow=c(1,3))

#Note this is a bagged tree since Im foring mytry "m" to equal 2

bag.adv<-randomForest( sales ~ TV+radio,data=Adver , subset=index ,
                       mtry=2,importance =TRUE,ntree=100)

yhat.bag = predict (bag.adv , newdata=test)
plot(yhat.bag , test$sales,main="Bagged Model",xlab="Predicted",ylab="Test Set Sales")
abline (0,1)

library(tree)
mytree<-tree(sales~TV+radio,train)
yhat.tree<-predict(mytree,newdata=test)
plot(yhat.tree,test$sales,main="Single Tree with 8 splits",xlab="Predicted",ylab="Test Set Sales")
abline(0,1)

mytree<-tree(sales~TV+radio,train,minsize=8,mindev=.0001)
yhat.tree<-predict(mytree,newdata=test)
plot(yhat.tree,test$sales,main="Single Tree with Deep Splits",xlab="Predicted",ylab="Test Set Sales")
abline(0,1)


#Lets take a look at the predicted surface of our bagged model
predictors<-data.frame(TV=rep(0:300,51),radio=rep(0:50,each=301))
bag.full<-randomForest( sales ~ TV+radio,data=Adver , subset=index ,
                        mtry=2,importance =TRUE,ntree=100)

pred.surface<-matrix(predict(bag.full,predictors),301,51)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,pred.surface,alpha=.4)



#The next code is not necessary for an analysis, but is helpful to bring home
#the points...
#1  Bagging improves prediction accuracy over a single tree just make sure 
#you resample at 50-100 times.
#2  The OOB error does a pretty good job of estimating what a true independent
#test set, its just a little bit too optimistic

OOB.MSE<-c()
test.MSE<-c()
for(i in 1:300){
  bag.adv<-randomForest( sales ~ TV+radio,data=Adver , subset=index ,
                         mtry=2,importance =TRUE,ntree=i) 
  
  yhat.bag = predict (bag.adv , newdata=test)
  test.MSE[i]<-mean((test$sales-yhat.bag)^2)
  OOB.MSE[i]<-mean((train$sales-bag.adv$predicted)^2)
}
par(mfrow=c(1,1))
plot(1:300,test.MSE,type="l",col="red",ylim=c(0,2.5))
lines(1:300,OOB.MSE,col="blue")
lines(1:300,rep(2.01,300),col="black",lty=2)
legend("topright",legend=c("Test MSE (Bagging)","OOB MSE (Bagging)","Test MSE (Single Tree)"),col=c("red","blue","black"),lty=c(1,1,2))







#Running the trees and the random forrest algorith for classification
#problems are easy enough.
#Just make sure your response is a "factor" so R knows that it IS a
#classification problem.  The rules of splitting the tree change since
#RSS no longer makes sense.

#Simple example to try to predict the Sales of 
attach(Carseats)
High=ifelse(Sales<=8,"No","Yes")
Carseats=data.frame(Carseats,High)
View(Carseats)

#Simple tree fit on training compared to test
set.seed(2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train,]
#Indepdent response to compare prediction performance on test set
High.test=High[-train]

tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)

#Overall Accuracy Rate
(86+57)/200

#Sensitivity
57/(57+27)

#Specificity
86/(86+30)

#Lets do a CV to prune the tree to see if we can do better and correct for overfitting
set.seed(3)
par(mfrow=c(1,1))
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)
plot(cv.carseats)
#Fit the pruned tree and visualize
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)

tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
#Accuracy
(94+60)/200
#Sensitivity
60/(60+24)
#Specificity
94/(94+22)

#We can verify the overfitting idea by saying to split the tree
#very deep
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
#Accuracy
(86+62)/200
#Sensitivity
62/(62+22)
#Specificity
86/(86+30)

#For ROC curves on a single decision tree you need predicted probabilities to
#use the previous R scripts.  Lets just use the example here with the last run.
tree.pred=predict(prune.carseats,Carseats.test,type="vector")
head(tree.pred)

library(ROCR)
pred <- prediction(tree.pred[,2], Carseats.test$High)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
#Note in the following code the term "train" means nothing here. 
#I'm just rinsing and repeating code the produces the curve.
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf,main="AUC of Test set of a Single Tree")
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Random forrest
#Here we will do a truly RF run by selecting mtry. mtry controls how many
#predictors are sampled for each bootstrap sample.
rf.car<-randomForest(High~.-Sales,Carseats,subset=train,mtry=5,importance=T,ntree=100)

#Making predictions on test and then observing accuracy rates
fit.pred<-predict(rf.car,newdata=Carseats.test,type="response")
table(fit.pred,Carseats.test$High) #Default prediction uses .5 as cut off you can change it specifying "cutoff" option
#Accuracy
(96+67)/(200)
#Sensitivity
67/(67+17)
#Specificity
96/(96+20)

#Go get the ROC
rf.pred<-predict(rf.car,newdata=Carseats.test,type="prob")
pred <- prediction(rf.pred[,2], Carseats.test$High)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
#Note in the following code the term "train" means nothing here. 
#I'm just rinsing and repeating code the produces the curve.
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf,main="AUC of Test set RF - mtry=5")
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#Which variables are important.  We can use variable importance.

varImpPlot (rf.car,type=1,main="Variable Importance")
varImpPlot (rf.car,type=2,main="Variable Importance")

attach(Carseats)
plot3d(Price,Advertising,Age,col=ifelse(High=="Yes","red","black"),size=4)

index1<-which(ShelveLoc=="Bad")
plot3d(Price[index1],Advertising[index1],Age[index1],col=ifelse(High=="Yes","red","black")[index1],size=4)

index2<-which(ShelveLoc=="Medium")
plot3d(Price[index2],Advertising[index2],Age[index2],col=ifelse(High=="Yes","red","black")[index2],size=4)

index3<-which(ShelveLoc=="Good")
plot3d(Price[index3],Advertising[index3],Age[index3],col=ifelse(High=="Yes","red","black")[index3],size=4)




#In case you are wondering more about k-nearest neighbors
#View the predictor space.
plot(radio,TV)

#knn visual. Use the "k" nearest neighbors in the predictor space to make the 
#prediction
plot(radio,TV,main="Predicting (30,150) \n k=4")
points(30,150,col="red",cex=6,pch=1)
points(30,150,col="red",cex=1,pch=19)


#The choice of k matters.  Larger k's have more smooth (yet less complex) predictions while
#small k predictions are more rigid and caputure every little detail that could potentially
#just be randomness.

#Lets look at k=5 predictions and again at k=25
#Note the input for the knn function is a little different and there are 2 different versions
#1 for regression and 1 for classification
predictors<-data.frame(TV=rep(0:300,51),radio=rep(0:50,each=301))
train<-data.frame(TV=TV,radio=radio)
knn.5<-FNN::knn.reg(train = train, test =predictors, y = sales, k = 5)

pred.surface<-matrix(knn.5$pred,301,51)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,pred.surface,alpha=.4)


predictors<-data.frame(TV=rep(0:300,51),radio=rep(0:50,each=301))
train<-data.frame(TV=TV,radio=radio)
knn.25<-FNN::knn.reg(train = train, test =predictors, y = sales, k = 25)

pred.surface<-matrix(knn.25$pred,301,51)
plot3d(TV,radio,sales)
surface3d(0:300,0:50,pred.surface,alpha=.4)


#To choose a k we can use a validation set or CV to decide.

#Cross Validation to select k
k=11  #number of folds

#This will be our matrix of cv test errors. 1 row for each fold, each column reperesents the number of predictors
#included in the model.
cv.errors<-matrix(NA,11,30)

set.seed(12345)
folds<-sample(1:k,nrow(Adver),replace=T)  #Here we are not forcing them to be as balanced as possible

folds<-c(rep(1:k,each=floor(nrow(Adver)/k)),1:(nrow(Adver)-floor(nrow(Adver)/k)*k)) #forcing more balanced
folds<-sample(folds,length(folds),replace=F) #shuffling


for(i in 1:k){
  train<-data.frame(TV=TV[folds!= i],radio=radio[folds != i])
  test<-data.frame(TV=TV[folds == i],radio=radio[folds == i])
  
  for(j in 1:30){
    knn<-FNN::knn.reg(train = train, test =test, y = sales[folds !=i], k = j)
    cv.errors[i,j]<-mean( ( sales[ folds ==i]-knn$pred)^2)
  }
}


cv.test<-apply(cv.errors,2,mean)
plot(1:30,cv.test,xlab="k",ylab="Test MSE", main="CV for selecting K in KNN",type="l",lwd=3)
points((1:30)[cv.test==min(cv.test)],min(cv.test),col="red" ,pch=19,cex=1.5 )

test.se<-apply(cv.errors,2,sd)/sqrt(k)
#Creating a 1-sd rule to decide # of predictors
sd1rule<-cv.test-test.se
index2<-which(sd1rule<min(cv.test))
points(index2,cv.test[index2],col="blue",pch=18)




