% Monte Carlo Simulation of the 2D phi^4 theory 
% with one real field component
% Tomasz Korzec 2021
function [e,s] = phi4MC(L, beta, lambda)
   Ntherm = 10000;
   Nprod  = 100000;
   
   np = [2:L,1];    % np(l) is the positive neighbor of l
   nm = [L, 1:L-1]; % nm(l) is the negative neighbor of l
   
   phi = ones(L,L); % initial configuration
   for l=1:Ntherm
      phi=sweep(beta, lambda, phi, np, nm);
   end
   
   s =zeros(Nprod,1);
   e =zeros(Nprod,1);

   for l=1:Nprod
      for k=1:10
         phi=sweep(beta, lambda, phi, np, nm);
      end
      s(l) = action(beta,lambda,phi,np,nm);
      e(l) = nncorr(phi,np);
   end
end

function phi = sweep(beta, lambda, phiold, np, nm)
   DPHI = 0.2; % interval for proposals
   phi=phiold;
   [Lx,Ly] = size(phi);
   for lx=1:Lx
   for ly=1:Ly
      phip = phi(lx,ly) + (rand()-0.5)*2*DPHI;
      % local magnetization
      b = phi(np(lx),ly)+phi(nm(lx),ly)+phi(lx,np(ly))+phi(lx,nm(ly));
      % action difference Snew-Sold
      DeltaS = phip^2-phi(lx,ly)^2 + ...
               lambda*((phip^2-1)^2-(phi(lx,ly)^2-1)^2) + ...
               beta*b*(phi(lx,ly)-phip);
      if rand<min(1,exp(-DeltaS))
         phi(lx,ly) = phip; 
      end
   end
   end
end

function s = action(beta,lambda,phi,np,nm)
% euclidean action of the 2D phi^4 model
   [Lx,Ly] = size(phi);
   s = 0;
   for lx=1:Lx
   for ly=1:Ly
      s = s + phi(lx,ly)^2+lambda*(phi(lx,ly)^2-1)^2;
      s = s - beta*phi(lx,ly)*(phi(np(lx),ly)+phi(lx,np(ly)));
   end
   end
end

function e = nncorr(phi,np,nm)
% nearest neighbor correlator
   [Lx,Ly] = size(phi);
   e = 0;
   for lx=1:Lx
   for ly=1:Ly
      e = e + phi(lx,ly)*(phi(np(lx),ly)+phi(lx,np(ly)));
   end
   end
   e = e/(2*Lx*Ly);
end
