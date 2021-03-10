library(ISLR)
dim(Auto)

newAuto<-Auto
#creating binary response for illustration
#Note: it is important to realise that this is for illustration purposes.  Converting
#a continuous response to binary typically is less advantageous. You are throwing information
#that could be used.
#MD's at hospitals love to do this, because it puts them in a comfortable place that most
#clinical research is presented in (contingency tables, and logistic regression with odds ratios)
newAuto$mpg<-factor(ifelse(Auto$mpg>median(Auto$mpg),"High","Low"))
newAuto$cylinders<-factor(newAuto$cylinders)
newAuto$origin<-factor(newAuto$origin)

#From here we are going to do a simple split of the data set and explor the training
#data set newAuto.  The test will be held out for assessment of the model fits.
set.seed(1234)
index<-sample(1:385,250,replace=FALSE)
test<-newAuto[-index,]
newAuto<-newAuto[index,]

#Explore the data in various ways

#For summary stats, one of the most important things to do 
#is make note if certain categorical predictors are highly unbalanced
#including the response.
#If predictors are highly unbalanced, cross validation runs later could yield 
#some errors during the run.  You might have to resort to a test/train for model building or
#a manual CV.  If the balance is extreme (85/15 or more) you may want to down sampling the larger
#group.


#
#aggregate is good for summary stats by groups for continous predictors
t(aggregate(weight~mpg,data=newAuto,summary))
t(aggregate(displacement~mpg,data=newAuto,summary))

#lets attach the training set newAuto so we don't have to keep writing newAuto$
attach(newAuto)
#Table of counts are helpful for categorcal predictors
ftable(addmargins(table(mpg,cylinders))) 
#It probably is wise to throw out the 3 and 5 cylinder ones or combine it with 
#four or six.  I'll remove to keep it short.
newAuto<-newAuto[-which(cylinders %in% c(3,5)),]
test<-test[-which(test$cylinders %in% c(3,5)),]

#Fixing the number of levels in cylinders
attach(newAuto)
cylinders=factor(cylinders)
levels(cylinders)
newAuto$cylinders<-factor(newAuto$cylinders)
#Doing the same to test
test$cylinders<-factor(test$cylinders)


ftable(addmargins(table(mpg,origin)))
ftable(addmargins(table(mpg,year)))
#Note year is continuous in the data set but it may be helpful 
#to convert to categorical. You just got to try and see.

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
pairs(newAuto[,3:6],col=mpg)
my.cor<-cor(newAuto[,3:6])

#If you have a lot of predictors, heatmap with correlations could
#be helpful to examine redundancy.
library(gplots)
library(ggplot2)
heatmap.2(my.cor,col=redgreen(75), 
          density.info="none", trace="none", dendrogram=c("both"), 
          symm=F,symkey=T,symbreaks=T, scale="none",key=T)
#Note we don't scale here because we are dealing with correlations that are essentially
#already scaled.

#Categorical predictors can suffer from "multicollinearity" as well. We can use previous
#graphics and table to see if the they are associated.
prop.table(table(origin,cylinders),2)
plot(origin~cylinders,col=c("purple","white","green"))
table(origin,cylinders)
#Here you can see that the cylinder 8, is confounded with origin. Knowing a car has
#8 cylinders, tells us that they must have origin 1.Origin 2 and 3 pretty much all
#have 4 cylinders.  If interpretation using logistic regression is part of the goal,
#take care here as the coefficients may not be interpretable if both predictors are used.


#This next section we will discuss in detail over the next few classes. These 
#tools are very handy for exploring many continous predictors at once.

#Another option here would be to do PCA among the continous predictors to see
#if the response naturally seperates out using PC's...or visualize with a heatmap.
pc.result<-prcomp(newAuto[,3:6],scale.=TRUE)
pc.scores<-pc.result$x
pc.scores<-data.frame(pc.scores)
pc.scores$mpg<-newAuto$mpg

#Scree plot
eigenvals<-(pc.result$sdev)^2
plot(1:4,eigenvals/sum(eigenvals),type="l",main="Scree Plot PC's",ylab="Prop. Var. Explained",ylim=c(0,1))
cumulative.prop<-cumsum(eigenvals/sum(eigenvals))
lines(1:4,cumulative.prop,lty=2)



#Use ggplot2 to plot the first few pc's
ggplot(data = pc.scores, aes(x = PC1, y = PC2)) +
  geom_point(aes(col=mpg), size=1)+
  ggtitle("PCA of Auto")
#So we can see some pretty good seperation here.


#From that it is building some confidence that a predictive model should
#perform pretty well.  So lets see an LDA and QDA fit and look 
#at a error rates on the test set.
library(ROCR)
library(MASS)
mylda<- lda(mpg ~ displacement+horsepower+weight+acceleration, data = newAuto)
myqda<- qda(mpg ~ displacement+horsepower+weight+acceleration, data = newAuto)

#confusion matrix
prd<-predict(mylda, newdata = test)$class
table(prd,test$mpg)

#Overall Misclassification Error rate on the test is 
14/(139)


#ROC
ldaprd<-predict(mylda, newdata = newAuto)$posterior
#correcting for the way lda creates predicted probabilities
ldaprd<-ldaprd[,2]

pred <- prediction(ldaprd, newAuto$mpg)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values

#Plot ROC
plot(roc.perf,main="LDA")
abline(a=0, b= 1) #Ref line indicating poor performance
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))



#Other ideas..  Depending on the problem, you may want to consider other cutpoints
#for the predicted probabilites for classification assignment.  Or the incorporation
#of priors.  Get creative and use some programming knowledge.








