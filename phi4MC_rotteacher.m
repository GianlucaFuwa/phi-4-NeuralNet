% Monte Carlo Simulation of the 2D phi^4 theory with one real field component
%
% Tomasz Korzec 2021
function [e,s,trainset] = phi4MC_rotteacher(L, M, beta, lambda, training)
   Ntherm = 10000;
   Nprod  = 50000;
   
   B = M; % 8x8 blocks for 'non-local' updates
   if L<B+2
      error('too small lattice')
   end
   % Indices of x1,x2,x3 inside of a (B+2) x (B+2) matrix
   % makes use of fact that matlab allows linear addressing of matrices
   Ix1 = [];
   Ix2 = [];
   for lx=2:B+1
   for ly=2:B+1
      l = (lx-1)*(B+2)+ly;
      if mod(lx+ly,2)==0
         Ix1 = [Ix1; l];
      else
         Ix2 = [Ix2; l];
      end
   end
   end
   Ix3 = [2:B+1, (B+1)*(B+2)+2:(B+2)*(B+2)-1, ... % vertical boundaries
          B+3:B+2:B*(B+2)+1, 2*(B+2):B+2:(B+1)*(B+2)]'; % horizontal boundaries
   bphi = zeros(B+2);
   x3mat = zeros(B,4);
   rotation = [3 1 0 2];
   % test whether indices are OK:
   
   %{
   Mx = zeros(B+2);
   Mx(Ix1) = 1;
   Mx(Ix2) = 2;
   Mx(Ix3) = 3;
   disp(Mx);
   %}  
      
   np = [2:L,1];    % np(l) is the positive neighbor of l
   nm = [L, 1:L-1]; %nm(l) is the negative neighbor of l
   
   % Ip helps performing operations modulo L
   for lx=1:2*L
   for ly=1:2*L
      Ip(lx,ly) = sub2ind([L,L],mod(lx-1,L)+1,mod(ly-1,L)+1);
   end
   end
   
   phi = ones(L,L); % initial configuration
   for l=1:Ntherm
      phi=sweep(beta, lambda, phi, np, nm);
   end
   
   s =zeros(Nprod,1);
   e =zeros(Nprod,1);
   
   data = zeros(B^2+4*B+2,Nprod);
   goals = zeros(B^2+4*B+2,Nprod);
   
   for l=1:Nprod
      for k=1:10
         phi=sweep(beta, lambda, phi, np, nm);
      end
      
      % create 'perfect' output for  a random block
      % 1) random corner of the block
      bx = randi(L);
      by = randi(L);
      % 2) cut out (B+2) x (B+2)  section of phi-field mod periodic b.c.
      bphi = phi(Ip(bx:bx+B+1, by:by+B+1));
      % 3) divide the block into x1,x2,x3 for the KI update
      
      % first half of internal points
      x1 = bphi(Ix1);
      % second half of internal points
      x2 = bphi(Ix2);
      % boundaries
      x3 = [bphi(Ix3); beta; lambda];
   
      % in a neural-net MC, blocksweeps gets replaced by a KI update,
      % and is followed by an accept-reject step
      
      [y1,y2] = blocksweeps(x1,x2,x3,Ix1,Ix2,Ix3);
      
      x3mat(:) = x3(1:4*B);
      x3matabssum = sum(abs(x3mat));
      x3matsum = sum(x3mat);
      [~,idx] = max(x3matabssum);
      rotxbphi = sign(x3matsum(idx))*rot90(bphi,rotation(idx));
      rotx1 = rotxbphi(Ix1);
      rotx2 = rotxbphi(Ix2);
      rotx3 = [rotxbphi(Ix3);beta;lambda];
      
      % place y1,y2 into the correct place of the block
      bphi(Ix1) = y1;
      bphi(Ix2) = y2;
      phi(Ip(bx:bx+B+1, by:by+B+1)) = bphi;
      
      rotybphi = sign(x3matsum(idx))*rot90(bphi,rotation(idx));
      roty1 = rotybphi(Ix1);
      roty2 = rotybphi(Ix2);
        
      if training == true
         data(:,l) = [rotx1;rotx2;rotx3];
         goals(:,l) = [roty1;roty2;rotx3];
      end   
      s(l) = action(beta,lambda,phi,np,nm);
      e(l) = nncorr(phi,np);
   end
   if training == true
    trainset = {}; 
    trainset{1} = data;
    trainset{2} = goals;
   end
end

function [y1, y2] = blocksweeps(x1,x2,x3,Ix1,Ix2,Ix3)
   
   DPHI = 0.2; % interval for proposals
   % reconstruct block-phi
   L = length(Ix3)/4+2;
   phi=zeros(L);
   phi(Ix1) = x1;
   phi(Ix2) = x2;
   phi(Ix3) = x3(1:end-2);
   beta = x3(end-1);
   lambda = x3(end);
   
   % block neighborhood relations
   np = [2:L,nan];
   nm = [nan, 1:L-1];
   
   % make 100 sweeps with Dirichlet b.c. 
   % to obtain an independent block-config
   for n=1:500 
      for lx=2:L-1
      for ly=2:L-1
         phip = phi(lx,ly) + (rand()-0.5)*2*DPHI;
         b = phi(np(lx),ly)+phi(nm(lx),ly)+phi(lx,np(ly))+phi(lx,nm(ly)); %local magnetization
         %action difference Snew-Sold
         DeltaS = phip^2-phi(lx,ly)^2 + lambda*((phip^2-1)^2-(phi(lx,ly)^2-1)^2) + ...
                  beta*b*(phi(lx,ly)-phip);
         if rand<min(1,exp(-DeltaS))
            phi(lx,ly) = phip; 
         end
      end
      end
   end
   y1 = phi(Ix1);
   y2 = phi(Ix2); 
end

function phi = sweep(beta, lambda, phiold, np, nm)
   DPHI = 0.2; % interval for proposals
   phi=phiold;
   [Lx,Ly] = size(phi);
   for lx=1:Lx
   for ly=1:Ly
      phip = phi(lx,ly) + (rand()-0.5)*2*DPHI;
      b = phi(np(lx),ly)+phi(nm(lx),ly)+phi(lx,np(ly))+phi(lx,nm(ly)); %local magnetization
      %action difference Snew-Sold
      DeltaS = phip^2-phi(lx,ly)^2 + lambda*((phip^2-1)^2-(phi(lx,ly)^2-1)^2) + ...
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
      s = s+ phi(lx,ly)^2+lambda*(phi(lx,ly)^2-1)^2;
      % add positive neighbor contributions
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
