L = 12;
M = 8;
beta = 0.6586246;
lambda = 0.25;
Nprod = 10000;

% create training set
%[~,~,trainsetrot] = phi4MC_rotteacher(L,M,beta,lambda,true);

% evaluate performance of Network in Simulation
[e,s,a] = phi4Net_rot(L,M,beta,lambda,Net,Nprod,18);
[me,de,dde,taue,dtaue]=UWerr(e);
%[ms,ds,dds,taus,dtaus]=UWerr(s./L^2/2); % action per link


