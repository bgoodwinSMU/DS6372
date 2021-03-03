
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
*1) Calculate summary statistics per response variable by group;
proc means data=Hemophil nmiss mean std stderr lclm uclm median min max qrange maxdec=2;
class Group;
var Activity;
run;

*2) Run single One Way ANOVA models for each variable.  
Perform residual diagnostics and make appropriate transformations if necessary to ensure each set of 
residuals meet the ANOVA assumption;
*Sort the data by group;
proc sort data=Hemophil;
by Group;
run;

*Test for normality;
proc univariate data=Hemophil normal;
by Group;
var Activity;
qqplot /normal (mu=est sigma=est);
run;
 
*Test for equality of variances and perform anova;
proc glm data=Hemophil;
class Group;
model Activity = Group;
means Group / hovtest=levene(type=abs) welch;
lsmeans Group /pdiff adjust=tukey plot=meanplot(connect cl) lines;
run;
quit;


*1) Calculate summary statistics per response variable by group;
proc means data=Hemophil nmiss mean std stderr lclm uclm median min max qrange maxdec=2;
class Group;
var Activity;
run;

*2) Run single One Way ANOVA models for each variable.  
Perform residual diagnostics and make appropriate transformations if necessary to ensure each set of 
residuals meet the ANOVA assumption;
*Sort the data by group;
proc sort data=Hemophil;
by Group;
run;

*Test for normality;
proc univariate data=Hemophil normal;
by Group;
var Antigen;
qqplot /normal (mu=est sigma=est);
run;
 
*Test for equality of variances and perform anova;
proc glm data=Hemophil;
class Group;
model Antigen = Group;
means Group / hovtest=levene(type=abs) welch;
lsmeans Group /pdiff adjust=tukey plot=meanplot(connect cl) lines;
run;
quit;

*3)Provide a scatterplot for Carrier vs Non ;
   proc sgplot data=Hemophil;
   title 'Carrier Vs Non Carrier';
   scatter x=Activity y=Antigen / group=Group;
run;

ods select Cov PearsonCorr;
proc corr data=Hemophil noprob outp=OutCorr /** store results **/
          nomiss /** listwise deletion of missing values **/
          cov;   /**  include covariances **/
var Group Activity Antigen;
run;
