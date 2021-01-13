*SAS code for Golf data set questions;

proc import datafile="/home/jturner1/Unit1/GolfData2.csv"
          dbms=dlm out=golf replace;
     delimeter=',';
     getnames=yes;
     
run;

proc print data=golf;
run;

data golf;
set golf;
logAvgWinnings=log(AvgWinnings);
run;



proc sgscatter data=golf;
  title "Scatterplot Matrix of Golf Variables";
  matrix logAvgWinnings Age AvgDrive DriveAcc	Greens	AvgPutts	Save;
run;

*M1;
*Running simple ols using glmselect on first set of named variables;
proc glmselect data = golf plots(stepaxis = number) = (criterionpanel ASEPlot) seed = 1;
partition fraction(test = .5); 
model logAvgWinnings=  Age AvgDrive DriveAcc	Greens	AvgPutts	Save   / selection = forward( stop = none);
run;

*M2;
*Running simple ols using glmselect on first set of named variables plus V12-V31;
proc glmselect data = golf plots(stepaxis = number) = (criterionpanel ASEPlot) seed = 1;
partition fraction(test = .5); 
model logAvgWinnings=  Age AvgDrive DriveAcc	Greens	AvgPutts	Save V12-V31  / selection = forward( stop = none);
run;



*M3;
*Running LASSO using glmselect on first set of named variables plus V12-V31;
proc glmselect data = golf plots(stepaxis = number) = (criterionpanel ASEPlot) seed = 1;
partition fraction(test = .5); 
model logAvgWinnings=  Age AvgDrive DriveAcc	Greens	AvgPutts	Save V12-V31  / selection = lasso( choose = cv stop = 27) CVDETAILS;
run;


*M4;
*Adding even more complexity;
*OLS;
proc glmselect data = golf plots(stepaxis = number) = (criterionpanel ASEPlot) seed = 1;
partition fraction(test = .5); 
model logAvgWinnings=  Age AvgDrive DriveAcc	Greens	AvgPutts	Age*Age Greens*Greens AvgPutts*AvgPutts AvgPutts*Greens V12-V31 / selection = forward( stop = none);
run;

*M5;
*Lasso;
proc glmselect data = golf plots(stepaxis = number) = (criterionpanel ASEPlot) seed = 1;
partition fraction(test = .5); 
model logAvgWinnings=  Age AvgDrive DriveAcc	Greens	AvgPutts	Age*Age Greens*Greens AvgPutts*AvgPutts AvgPutts*Greens V12-V31 / selection = lasso( choose = cv stop = 31) CVDETAILS;
run;




