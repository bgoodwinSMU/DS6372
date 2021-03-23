#The following script is structured into 3 sections.
#The first is a data reading in and EDA phase, the second is interpertation using logistic regression and building up things,
#and the third illustrates Lasso and provides some of your first ROC curves that we will discuss at the beginning of next class.

#DATA entry and EDA.  You guys are familiar with this data set already.  Its the same as the PCA Unit 9 code.
library(ISLR)
dim(Auto)

newAuto<-Auto
#creating binary response for illustration
newAuto$mpg<-factor(ifelse(Auto$mpg>median(Auto$mpg),"High","Low"))
newAuto$cylinders<-factor(newAuto$cylinders)
newAuto$origin<-factor(newAuto$origin)
newAuto$mpg<-relevel(newAuto$mpg, ref = "Low")  #""Yes/1" is high mpg "No/0" is low mpg.  This forces the odds to be what you want it to be.

#Explore the data in various ways

#For summary stats, one of the most important things to do 
#is make note if certain categorical predictors are highly unbalanced
#including the response.
#If predictors are highly unbalanced, cross validation runs later could yield 
#some errors during the run.  You might have to resort to a test/train for model building or
#a manual CV.

#
#aggregate is good for summary stats by groups for continous predictors
aggregate(weight~mpg,data=newAuto,summary)
aggregate(displacement~mpg,data=newAuto,summary)


#lets attach newAuto so we don't have to keep writing newAuto$
attach(newAuto)
#Table of counts like proc freq are helpful for categorcal predictors
ftable(addmargins(table(mpg,cylinders))) 
#It probably is wise to throw out the 3 and 5 cylinder ones or combine it with 
#four or six.  I'll remove to keep it short.
newAuto<-newAuto[-which(cylinders %in% c(3,5)),]
attach(newAuto)
cylinders=factor(cylinders)
levels(cylinders)

ftable(addmargins(table(mpg,origin)))
ftable(addmargins(table(mpg,year))) 

#to get proportions that make sense
prop.table(table(mpg,cylinders),2)
prop.table(table(mpg,origin),2)
prop.table(table(mpg,year),2)

#Visualize
plot(mpg~cylinders,col=c("red","blue"))
plot(mpg~origin,col=c("red","blue"))
plot(mpg~year,col=c("red","blue"))

#Visualize
plot(weight~mpg,col=c("red","blue"))
plot(acceleration~mpg,col=c("red","blue"))
plot(displacement~mpg,col=c("red","blue"))

#Examine the correlation between the continous predictors
pairs(newAuto[,3:6])
my.cor<-cor(newAuto[,3:6])
my.cor
pairs(newAuto[,3:6],col=mpg)

#If you have a lot of predictors, heatmap with correlations could
#be helpful to examine redundancy.
library(gplots)
library(ggplot2)
heatmap.2(my.cor,col=redgreen(75), 
          density.info="none", trace="none", dendrogram=c("row"), 
          symm=F,symkey=T,symbreaks=T, scale="none")
#Note we don't scale here because we are dealing with correlations that are already
#scaled.
#Note above. You can also use the plots previously to examine one by one
#if predictors are associated category to categor or category to continuous.

#Another option here would be to do PCA among the continous predictors to see
#if they seperate out.  Or a heatmap.
pc.result<-prcomp(newAuto[,3:6],scale.=TRUE)
pc.scores<-pc.result$x
pc.scores<-data.frame(pc.scores)
pc.scores$mpg<-newAuto$mpg

#Use ggplot2 to plot the first few pc's
ggplot(data = pc.scores, aes(x = PC1, y = PC2)) +
  geom_point(aes(col=mpg), size=1)+
  ggtitle("PCA of Auto")
#So we can see some pretty good seperation here.



#Another approach we will see in unit 13 is to cluster the data in spepcifc ways. I will illustrate this with
#a new data set next class.








#Here is where things get really personalized depending on the goal.  I've seperated these out into "Interpretation"
#and "Prediction Only" tasks.  Some things are not covered entirely and we will discuss them next week.



##############################
#Interpretation
#The purpose of this code is to illustrate some basic functionality of logistic regression in R.
#Some of the continuous variables look redundant, but for start we will just include everything.
newAuto<-na.omit(newAuto)
model.main<-glm(mpg ~ cylinders+displacement+horsepower+weight+acceleration+year, data=newAuto,family = binomial(link="logit"))
library(ResourceSelection)
library(car)

#Using this tool, GVIF is the same as VIF for continuous predictors only
#For categorical predictors, the value GVIG^(1/(2*df)) should be squared and interpreted
#as a usuaul vif type metric.The following code can be used to interpret VIFs like we 
#discussed in class.
(vif(model.main)[,3])^2

#As expected displacement and the other continuous variables have moderately high VIF.  Based on the pairwise scatterplots
#I believe it is clear.  It is also apparent that cylinders is associated with some of the predictors
#as well. Higher cylinders tends to produce higher horsepower. 
#Remember we should not only resort to things like VIF, we should look at the output and 
#see if things make sense.
attach(newAuto)
prop.table(table(mpg,cylinders),2)
t(aggregate(weight~cylinders,data=newAuto,summary))
t(aggregate(acceleration~cylinders,data=newAuto,summary))
t(aggregate(horsepower~cylinders,data=newAuto,summary))
t(aggregate(displacement~cylinders,data=newAuto,summary))
#Ask yourself what this information is saying.  Incorporate that type of knowledge in your project.

#Hosmer Lemeshow test for lack of fit.  Use as needed.  The g=10 is an option that deals with the continuous predictors if any are there.
#This should be increased with caution. 
hoslem.test(model.main$y, fitted(model.main), g=10)


#Summary of current fit
summary(model.main)
#I'm not aware of a nice little automated way to produce Odds ratio metrics
#like SAS does.  Using the summary coefficients we can generate CI for each one in the table
exp(cbind("Odds ratio" = coef(model.main), confint.default(model.main, level = 0.95)))
#

#READ ME HERE  VERY IMPORTANT
#If you dig into the output (cylinder 8 for example) you will notice some coefficients dont make sense compared 
#to exploratory work.  The odds of having high mpg for 8 cylinder is 16 times that of a 4 cylinder?

#This is due to the fact that cylinders are correlated with everything.  Go back to EDA and verify.  We just don't
#see the VIF's look too suspect.  
t(aggregate(horsepower~cylinders,data=newAuto,summary))
plot(horsepower~cylinders,data=newAuto)
#Recall the interpretation of regression coefficients, the effect (odds ratio) is describing the
#effect of one predictor, holding the other variables fixed.  So an ODDS ratio of 10 for cylinder 8
#compared to cylinder 4 is comparing two cars (one 8 cyl one 4 cylinder but with similar acceleration,
#weight, horsepower, etc.  You can't hold those fixed in this case with a car.


#For this scenario it might be helpful to just manually fit some models first.
#If one were to conduct forward selection (see below (NO cv/ test set, just AIC selected)), R would want to keep all of the 
#highly correlated predictors in questions and the same interpretation problem occurs.
model.null<-glm(mpg ~ 1, data=newAuto,family = binomial(link="logit"))


#This starts with a null model and then builds up using forward selection up to all the predictors that were specified in my
#main model previously.
step(model.null,
     scope = list(upper=model.main),
     direction="forward",
     test="Chisq",
     data=newAuto)



#To deal with the redundamcy, I would throw the cylinder variable out and then see what happens
model.main<-glm(mpg ~ displacement+horsepower+weight+acceleration, data=newAuto,family = binomial(link="logit"))
summary(model.main)
exp(cbind("Odds ratio" = coef(model.main), confint.default(model.main, level = 0.95)))
vif(model.main)


#Residual diagnostics can be obtained using
plot(model.main)
#The only plot worth examining here is the fourth/final one that allows you to examine levaeage and cooks d.  You read this 
#just like the MLR one.



#With a simplistic model with no lack of fit issues, we can beging providing statistical inference if no
#interactions are present
summary(model.main)
#I'm not aware of a nice little automated way to produce Odds ratio metrics
#like SAS does.  Using the summary coefficients we can generate CI for each one in the table
exp(cbind("Odds ratio" = coef(model.main), confint.default(model.main, level = 0.95)))



#Playing around with adding more complexity to see if anything sticks compared to the simpler model
model.complex<-glm(mpg ~ displacement+horsepower+weight+acceleration+horsepower:weight+displacement:horsepower+weight:acceleration, data=newAuto,family = binomial(link="logit"))
step(model.main,
     scope = list(upper=model.complex),
     direction="forward",
     test="Chisq",
     data=newAuto)
hoslem.test(model.complex$y, fitted(model.complex), g=10)
#This feature selection included some of the interaction terms.  However their coeffcients are quite small, (see below *10^-4) so we might
#consider this model in terms of helping predictions out, but for interpretation, its going to be more challenging describing effects
#in a more complicated way, but in the end will still look very similar.

summary(glm(mpg ~ displacement+horsepower+weight+acceleration+displacement:horsepower+weight:acceleration, data=newAuto,family = binomial(link="logit")))


#The script above does not provide LASSO  calls, but I've included it in the next set of examples.  







#############################
#Prediction only
#With just one data set, to assess prediction preformance using Logistic and do feature selection
#properly, we should run some sort of CV.  We can either use LASSO to do this
#or the bestglm package which has forward selection techniques.
#The bestglm doesn't have a graphic unfortunately.  We could also write a script
#to do CV ourself or do a train/test split.

#The following code runs the CV as if there is no test set
#You can rinse and repeat the prediction and ROC curves with a new data set
#easily.  More examples coming in Unit 13 and Unit 14.

library(glmnet)
library(bestglm)
dat.train.x <- model.matrix(mpg~cylinders+displacement+horsepower+weight+acceleration+year+origin-1,newAuto)
dat.train.y<-newAuto[,1]
library(glmnet)
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

#Get training set predictions...We know they are biased but lets create ROC's.
#These are predicted probabilities from logistic model  exp(b)/(1+exp(b))
fit.pred <- predict(finalmodel, newx = dat.train.x, type = "response")

#Note that here above you can use thepredictions to create a confusion matrix if so desired.  I'm leaving this for groups to sort out.
#Probably will see an example in the future units.


#Create ROC curves (Remember if you have a test data set, you can use that to compare models)
library(ROCR)
pred <- prediction(fit.pred[,1], dat.train.y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values

#Plot ROC
plot(roc.perf,main="LASSO")
abline(a=0, b= 1) #Ref line indicating poor performance
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))



#In addition to LASSO, if we are concerned that the biased estiamtes
#are affecting our model, we can go back and refit using regular 
#regression removing the variables that have no importance.
coef(finalmodel)

olog<-glm(mpg~cylinders+horsepower+weight+year+origin,data=newAuto,family=binomial)
fit.pred <- predict(olog, newx = dat.train.x, type = "response")

pred <- prediction(fit.pred, dat.train.y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values

#Plot ROC
plot(roc.perf,main="Ordingary Logistic")
abline(a=0, b= 1) #Ref line indicating poor performance
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))





