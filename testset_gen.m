L = 12;
M = 8;
beta = 0.6586246;
lambda = 0.25;
Nprod = 10000;

%create training set
%[~,~,trainsetrot] = phi4MC_rotteacher(L,M,beta,lambda,true);

%test NN on exam
[e,s,a] = phi4MC_rotexam(L,M,beta,lambda,Adam8_400rotS,Nprod,18);
[me,de,dde,taue,dtaue]=UWerr(e);
%[ms,ds,dds,taus,dtaus]=UWerr(s./L^2/2); % action per link


