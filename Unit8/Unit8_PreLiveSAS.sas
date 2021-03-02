
Title1 "Hemophilia Data";
Proc Format;
 Value Groups 1="Noncarriers" 2="Carriers";
Run;
Data Hemophil;
 Input Group Activity Antigen;
 Format Group Groups.;
 Label Group="Group" Activity="AHF Activity" Antigen="AHF Antigen";
Datalines;
  1  -0.0056  -0.1657
  1  -0.1698  -0.1585
  1  -0.3469  -0.1879
  1  -0.0894   0.0064
  1  -0.1679   0.0713
  1  -0.0836   0.0106
  1  -0.1979  -0.0005
  1  -0.0762   0.0392
  1  -0.1913  -0.2123
  1  -0.1092  -0.1190
  1  -0.5268  -0.4773
  1  -0.0842   0.0248
  1  -0.0225  -0.0580
  1   0.0084   0.0782
  1  -0.1827  -0.1138
  1   0.1237   0.2140
  1  -0.4702  -0.3099
  1  -0.1519  -0.0686
  1   0.0006  -0.1153
  1  -0.2015  -0.0498
  1  -0.1932  -0.2293
  1   0.1507   0.0933
  1  -0.1259  -0.0669
  1  -0.1551  -0.1232
  1  -0.1952  -0.1007
  1   0.0291   0.0442
  1  -0.2280  -0.1710
  1  -0.0997  -0.0733
  1  -0.1972  -0.0607
  1  -0.0867  -0.0560
  2  -0.3478   0.1151
  2  -0.3618  -0.2008
  2  -0.4986  -0.0860
  2  -0.5015  -0.2984
  2  -0.1326   0.0097
  2  -0.6911  -0.3390
  2  -0.3608   0.1237
  2  -0.4535  -0.1682
  2  -0.3479  -0.1721
  2  -0.3539   0.0722
  2  -0.4719  -0.1079
  2  -0.3610  -0.0399
  2  -0.3226   0.1670
  2  -0.4319  -0.0687
  2  -0.2734  -0.0020
  2  -0.5573   0.0548
  2  -0.3755  -0.1865
  2  -0.4950  -0.0153
  2  -0.5107  -0.2483
  2  -0.1652   0.2132
  2  -0.2447  -0.0407
  2  -0.4232  -0.0998
  2  -0.2375   0.2876
  2  -0.2205   0.0046
  2  -0.2154  -0.0219
  2  -0.3447   0.0097
  2  -0.2540  -0.0573
  2  -0.3778  -0.2682
  2  -0.4046  -0.1162
  2  -0.0639   0.1569
  2  -0.3351  -0.1368
  2  -0.0149   0.1539
  2  -0.0312   0.1400
  2  -0.1740  -0.0776
  2  -0.1416   0.1642
  2  -0.1508   0.1137
  2  -0.0964   0.0531
  2  -0.2642   0.0867
  2  -0.0234   0.0804
  2  -0.3352   0.0875
  2  -0.1878   0.2510
  2  -0.1744   0.1892
  2  -0.4055  -0.2418
  2  -0.2444   0.1614
  2  -0.4784   0.0282
;

*Your assignment code goes here.;






*Once the residuals look good one at a time. We need to test for equal covariance matrices as this is another assumption.;
*In general, we can not visually assess this so we will rely on a test.;
*The output from this proc will be quite large, you just need to find the table that explicitly tests for equal covariance matrices.;
*If the the test is rejected, then we conclude that the covariance matrices are indeed different and report that analysis using this
*method is flawed.   If we fail to reject, then we are comfortable (as good as we can be) that the multivariate assumptions for MANOVA are okay and we
*move on to hypothesis testing.  Note: If any variables were transformed make sure those transformed variables are used in the steps moving forward.;

proc discrim data=Hemophil pool=test;
class group;
var Activity Antigen;
run;

*If we failed to reject the equal covariance matrices test, then we can move forward and conduct the MANOVA F-test.;
*This test will tell us if there are any differences between the groups for any of the variables.;
*The only thing you need from this output is the big MANOVA ftest table that has the Wilk's Lambda, Pillai's Trace, etc statistics.;


Proc GLM Data=Hemophil plot=diagnostics;
 Class Group;
 Model Activity Antigen = Group;
 Manova H=_All_ / PrintE PrintH Canonical;
Run;






*If the MANOVA F test is rejected, then we can go back and dive into our univariate analysis from before;
*We just need to penalize ourself for multiple testing.  In this case, there are only two groups Carries versus non carrries and two variables Antigen
*and activity, so there are technically only two tests we need to conduct.  The bonferroni adjusted test would just be to use a significance cut off
*of .05/2 or .025.  Note how I included that adjustment with the alpha option.;


Proc glm Data=Hemophil plot=diagnostics;
 Class Group;
 Model Activity = Group;
 lsmeans Group / pdiff CLDIFF T Alpha=0.025 ;
 Run;


Proc glm Data=Hemophil plot=diagnostics;
 Class Group;
 Model Activity = Group;
 lsmeans Group / pdiff CLDIFF T Alpha=0.025 ;
 Run;


*From these analysis, summaries of the  hypothesiss can be reported describing for which variables and which groups have statistically different means.;






*To run an LDA analysis, the first two steps are exactly the same as the MANOVA portion above.;
*Plot data, get summary statistics, and examine the residuals from running ANOVA for each variable.;
*After that we go strait to proc discrim to run the LDA analysis. Suppose that in addition to running LDA I have 3 new observations
for Activiry and Antigen that I want to classify as either being a Carrier or Non carrier.  I've put these 3 new observations in a data set
called "myprediction";

data myprediction;
input Activity Antigen;
cards;
-.2 -.2
-.2 0
-.3 0
0 .1;
run;


*The following code will Run LDA or QDA depending on the test for equal covariance matrices(just like in the MANOVA analysis).
*Options pool=test and crossvalidate need to be given. We include the data set "myprediction" as a test data set to predict these new subjects
as either Carriers or noncarriers.;


proc discrim data=Hemophil pool=test crossvalidate  testdata=myprediction list testlist;
class group;
var Activity Antigen;
priors "Carriers"=.5 "Noncarriers"=.5;
run;

*SAS default for priors is to assume that the groups are equally likely to occur.  If you know that is not true and have a better understanding of
how likely these groups would occur based on a random picking a person, you could incorporate that knowledge instead.;


*The key outputs for this proc will be looking at the "Linear Discriminant Function for Group" table as this will allow you to report a formula to give
*to others to help them to predict future observations (see slide 45). The other main table will be to examine the classification summary table (see slide 46)
to see how well the LDA was able to predict the groups using cross validation.;
*There is also a table that provides the predictors for the new observations we included in the test data set;


















