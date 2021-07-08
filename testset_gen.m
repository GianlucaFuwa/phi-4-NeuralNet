L = 12;
M = 8;
beta = 0.6586246;
%beta = 0.61567;
lambda = 0.25;
%lambda = 0.5;
Nprod = 10000;

%create training set
%[~,~,trainset] = phi4MC_teacher(L,M,beta,lambda,true);
%[e,s] = phi4MC(L,beta,lambda,100000);
%[~,~,trainsetdct] = phi4MC_dctteacher(L,M,beta,lambda,true);
%[e,s,trainsetrot] = phi4MC_rotteacher(L,M,beta,lambda,true);

%test NN on exam
%[e,s,a] = phi4MC_exam(L,M,beta,lambda,SGDM8_400W2H,Nprod,18);
%[e,s,a] = phi4MC_Netonly(L,M,beta,lambda,NetAdam4_500,Nprod);
%[e,s,a] = phi4MC_dctexam(L,M,beta,lambda,NetSGDM8_400_10kdctL1,Nprod,18);
[e,s,a] = phi4MC_rotexam(L,M,beta,lambda,Adam8_400rotS,Nprod,18);
[me,de,dde,taue,dtaue]=UWerr(e);
%[ms,ds,dds,taus,dtaus]=UWerr(s./L^2/2); % action per link
%[mmL,dmL,ddmL,taumL,dtaumL]=UWerr(mL);


