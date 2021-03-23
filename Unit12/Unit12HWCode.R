library(ISLR)
newAuto<-Auto
#creating binary response for illustration
#Note: it is important to realise that this is for illustration purposes.  Converting
#a continuous response to binary typically is less advantageous. You are throwing information
#that could be used.
#MD's at hospitals love to do this, because it puts them in a comfortable place that most
#clinical research is presented in (contingency tables, and logistic regression with odds ratios)

#Creating a categorical response for illustrative purposes.
newAuto$mpg<-factor(ifelse(Auto$mpg>median(Auto$mpg),"High","Low"),levels=c("Low","High"))  #last level is the success

newAuto$cylinders<-factor(newAuto$cylinders)
#Removing data with 3 and 5 cylinders and making sure are R treats the predictor with just 3 levels and not 5.
newAuto<-newAuto[-which(newAuto$cylinders %in% c(3,5)),]
newAuto$cylinders<-factor(newAuto$cylinders)
#Creating just two origins by combining 2 and 3 again for keeping things simple.
newAuto$origin[which(newAuto$origin==3)]<-2
newAuto$origin<-factor(newAuto$origin)

#From here we are going to do a simple split of the data set and explore the training
#data set newAuto.  The test will be held out for assessment of the model fits.
set.seed(1234)
index<-sample(1:385,250,replace=FALSE)
test<-newAuto[-index,-9]
train<-newAuto[index,-9]


#Explore the data in various ways

#For summary stats, one of the most important things to do 
#is make note if certain categorical predictors are highly unbalanced
#including the response.
#If predictors are highly unbalanced, cross validation runs later could yield 
#some errors during the run.  You might have to resort to a test/train for model building or
#a manual CV.  If the balance is extreme (85/15 or more) you may want to down sample the larger
#group.


#A great overview for a reasonable number of predictors is the scatter plot matrix ussing GGally.  We can see alot of natural trends in the data.

library(GGally)
ggpairs(newAuto,columns=2:8,aes(colour=mpg))

#If we want to make things more concise we can make specific graphics.
attach(newAuto)
prop.table(table(mpg,cylinders),2)
plot(mpg~cylinders,col=c("red","blue"))

t(aggregate(weight~mpg,data=newAuto,summary))
plot(weight~mpg,col=c("red","blue"))

# Don't forget PCA can be used to give us a similar sense of what is going on with continuous stuff along with heatmaps with cluster results.
#Also basic summar statistics should be reported as well that I'm omitting to keep this manuscript shorter.

#Question 1.  Using the summary statistics provided or some of your own, briefly describe if any of the predictors look like they could be helpful in predicting the mpg categories.

#Question 2.  #We discussed previously that multicollinearity can happen among categorical as well as continuous predictors.  Our focuse has been on exploration of the continuous variables.  For example, create a boxplot of weight by cylinders.  If there is no multicollinearity then the average weight should not depend on the cylnder of the car.  What does your graph suggest?  What about origin and cylinders are they "correlated" visually?    
#Answer
prop.table(table(origin,cylinders),2)
plot(origin~cylinders,col=c("purple","green"))
table(origin,cylinders)


#Simple logistic regression connections with 2x2 table.
library(epitools)
mymat<-table(origin,mpg)
mymat
oddsratio.wald(mymat)


#Question 3
simple.log<-glm(mpg~origin,family="binomial",data=newAuto)
summary(simple.log)
exp(2.728)



#Feature selection using Lasso and Stepwise.
library(MASS)
library(tidyverse)
library(cars)
full.log<-glm(mpg~.,family="binomial",data=train)
step.log<-full.log %>% stepAIC(trace=FALSE)

#Lets say we would like to interpret the step.log model
summary(step.log)
exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)  #Last column is interpretted based off of rule of 5 or 10

#Question 4.  Using the stepwise model, provide interpretation on how weight and origin impact the mpg status.


#Making a LASSO call is essentially the same as before.
library(glmnet)
dat.train.x <- model.matrix(mpg~cylinders+displacement+horsepower+weight+acceleration+year+origin-1,train)
dat.train.y<-train[,1]
cvfit <- cv.glmnet(dat.train.x, dat.train.y, family = "binomial", type.measure = "class", nlambda = 1000)
plot(cvfit)
coef(cvfit, s = "lambda.min")
#CV misclassification error rate is little below .1
cvfit$cvm[which(cvfit$lambda==cvfit$lambda.min)]

#Optimal penalty
cvfit$lambda.min

#For final model predictions go ahead and refit lasso using entire
#data set
finalmodel<-glmnet(dat.train.x, dat.train.y, family = "binomial",lambda=cvfit$lambda.min)




#Comparing prediction performance
#Lets compare the prediction performance of the stepwise and lasso models on the test set.

#These are predicted probabilities from the logistic model  exp(b)/(1+exp(b))
dat.test.x<-model.matrix(mpg~cylinders+displacement+horsepower+weight+acceleration+year+origin-1,test)
fit.pred.lasso <- predict(finalmodel, newx = dat.test.x, type = "response")
fit.pred.step<-predict(step.log,newdata=test,type="response")

test$mpg[1:15]
fit.pred.lasso[1:15]


#Lets use the predicted probablities to classify the observations and make a final confusion matrix for the two models.  We can use it to calculate error metrics.
#Lets us a cutoff of 0.5 to make the classification.
cutoff<-0.5
class.lasso<-factor(ifelse(fit.pred.lasso>cutoff,"High","Low"),levels=c("Low","High"))
class.step<-factor(ifelse(fit.pred.step>cutoff,"High","Low"),levels=c("Low","High"))

#Confusion Matrix for Lasso
conf.lasso<-table(class.lasso,test$mpg)


conf.step<-table(class.step,test$mpg)


#From the tables we can calculate the overall accuracy.
sum(diag(conf.lasso))/sum(conf.lasso)

sum(diag(conf.step))/sum(conf.step)



#Rather than making the calculations from the table, we can compute them more quickly using the following code which just checks if the prediction matches the truth and then computes the proportion.
mean(class.lasso==test$mpg)
mean(class.step==test$mpg)




#Sensitivity and Specificity discussion.
conf.step

sensitivity<-69/72
specificity<-59/63

sensitivity
specificity





cutoff<-0.1
class.step<-factor(ifelse(fit.pred.step>cutoff,"High","Low"),levels=c("Low","High"))
conf.step<-table(class.step,test$mpg)
conf.step

#Accuracy
mean(class.step==test$mpg)

sensitivity<-71/72
specificity<-52/63
sensitivity
specificity




#ROC curves
library(ROCR)
results.lasso<-prediction(fit.pred.lasso, test$mpg,label.ordering=c("Low","High"))
roc.lasso = performance(results.lasso, measure = "tpr", x.measure = "fpr")
plot(roc.lasso,colorize = TRUE)
abline(a=0, b= 1)


results.step<-prediction(fit.pred.step, test$mpg,label.ordering=c("Low","High"))
roc.step = performance(results.step, measure = "tpr", x.measure = "fpr")
plot(roc.step,colorize = TRUE)
abline(a=0, b= 1)

simple.log<-glm(mpg~origin+horsepower,family="binomial",data=train)
fit.pred.origin<-predict(simple.log,newdata=test,type="response")
results.origin<-prediction(fit.pred.origin,test$mpg,label.ordering=c("Low","High"))
roc.origin=performance(results.origin,measure = "tpr", x.measure = "fpr")

plot( roc.lasso)
plot(roc.step,col="orange", add = TRUE)
plot(roc.origin,col="blue",add=TRUE)
legend("bottomright",legend=c("Lasso","Stepwise","Origin/Horsepower Only"),col=c("black","orange","blue"),lty=1,lwd=1)
abline(a=0, b= 1)

