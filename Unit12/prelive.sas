 data coronary;
  input sex ecg age ca @@  ;
  cards;
  0 0 28 0   1 0 42 1    0 1 46 0  1 1 45 0
  0 0 34 0   1 0 44 1    0 1 48 1  1 1 45 1
  0 0 38 0   1 0 45 0    0 1 49 0  1 1 45 1
  0 0 41 1   1 0 46 0    0 1 49 0  1 1 46 1
  0 0 44 0   1 0 48 0    0 1 52 0  1 1 48 1
  0 0 45 1   1 0 50 0    0 1 53 1  1 1 57 1
  0 0 46 0   1 0 52 1    0 1 54 1  1 1 57 1
  0 0 47 0   1 0 52 1    0 1 55 0  1 1 59 1
  0 0 50 0   1 0 54 0    0 1 57 1  1 1 60 1
  0 0 51 0   1 0 55 0    0 2 46 1  1 1 63 1
  0 0 51 0   1 0 59 1    0 2 48 0  1 2 35 0
  0 0 53 0   1 0 59 1    0 2 57 1  1 2 37 1
  0 0 55 1   1 1 32 0    0 2 60 1  1 2 43 1
  0 0 59 0   1 1 37 0    1 0 30 0  1 2 47 1
  0 0 60 1   1 1 38 1    1 0 34 0  1 2 48 1
  0 1 32 1   1 1 38 1    1 0 36 1  1 2 49 0
  0 1 33 0   1 1 42 1    1 0 38 1  1 2 58 1
  0 1 35 0   1 1 43 0    1 0 39 0  1 2 59 1
  0 1 39 0   1 1 43 1    1 0 42 0  1 2 60 1
  0 1 40 0   1 1 44 1
  ;
run;quit;

*Explore;
proc freq data=coronary;
tables sex*ca ecg*ca  sex*ecg*ca / chisq relrisk;
run;quit;

proc means data=coronary;
class ca sex ecg;
types ca ca*sex ca*ecg;
var age;
run;

*Simple proc logistic call to play around with;
proc logistic data=coronary ;
class sex age ecg / param=ref;
model ca(event='1')= sex/ scale=none aggregate influence lackfit;
run;

proc glm data = coronary;
    class sex ca;
    model sex = ca;
run;quit;

*Simple proc logistic call to play around with;
proc logistic data=coronary ;
class sex / param=ref;
model ca(event='1')= sex/ scale=none aggregate influence lackfit;
run;

