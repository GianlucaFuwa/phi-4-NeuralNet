L = 12;
M = 8;
beta = 0.6586246;
lambda = 0.25;
Nprod = 2000000;

% create training set
%[~,~,trainsetrot] = phi4MC_rotteacher(L,M,beta,lambda,true);

% evaluate performance of Network in Simulation
[e,s,G0,chi2,a] = phi4Net_rot(L,M,beta,lambda,Net,Nprod,floor(L^2/M^2),2);
[me,de,dde,taue,dtaue]=UWerr(e); % energy
%[ms,ds,dds,taus,dtaus]=UWerr(s./L^2/2); % action per link
%[mG0,dG0,ddG0,tauG0,dtauG0]=UWerr(G0); % G(0)
%[mchi2,dchi2,ddchi2,tauchi2,dtauchi2]=UWerr(chi2); % susceptibility


