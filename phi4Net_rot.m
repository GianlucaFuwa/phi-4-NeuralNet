% Monte Carlo Simulation of the 2D phi^4 theory with one real field component
%
% Tomasz Korzec 2021
function [e,s,a] = phi4MC_rotexam(L, M, beta, lambda, net, nprod, cycles)
   Ntherm = 1000;
   Nprod  = nprod;
   Acc = 0;
   Delta = 0;
   Loccylces = 2;
   Netcycles = cycles;
   
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
   x3mat = zeros(B,4);
   rotation = [3 1 0 2];
      
   np = [2:L,1];    % np(l) is the positive neighbor of l
   nm = [L, 1:L-1]; %nm(l) is the negative neighbor of l
   
   % Ip helps performing operations modulo L
   Ip = zeros(2*L);
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
   
   for l=1:Nprod
      for k=1:Loccylces
         phi=sweep(beta, lambda, phi, np, nm);
      end
      for i=1:Netcycles
      % create 'perfect' output for  a random block
      % 1) random corner of the block
      bx = randi(L);
      by = randi(L);
      % 2) cut out (B+2) x (B+2)  section of phi-field mod periodic b.c.
      bphi = phi(Ip(bx:bx+B+1, by:by+B+1));
      % 3) divide the block into x1,x2,x3 for the KI update
     
      % boundaries
      x3 = bphi(Ix3);
      
      Sold = block_action(beta,lambda,bphi,np,B);
      
      % find one of 4 boundaries with highest  sum(|phi(x)|) and rotate so
      % it's on top. Then multiply so sum(phi(x)) of that boundary is
      % positive
      x3mat(:) = x3; % pass 4 boundaries onto a (4xM) matrix
      x3matabssum = sum(abs(x3mat)); % calc abs. sums of each column 
      [~,idx] = max(x3matabssum); % find column with highest abs sum      
      x3sum = sum(x3mat(:,idx)); % calc normal sum of found boundary
      % rotate block and multiply by the sign of sum of the boundary vector
      bphi = sign(x3sum)*rot90(bphi,rotation(idx)); 
      
      % generate input vectors for Neural Network
      rotx1 = bphi(Ix1);
      rotx2 = bphi(Ix2);
      rotx3 = [bphi(Ix3);beta;lambda]; 
      
      % in a neural-net MC, blocksweeps gets replaced by a KI update,
      % and is followed by an accept-reject step
      if rand<0.5
      net = ffnet_forward(net,[rotx1;rotx2;rotx3],B);
      bphi(Ix1)  = net.O(1:B^2/2);
      bphi(Ix2)  = net.O(B^2/2+1:B^2);
      else 
      net = ffnet_backward(net,[rotx1;rotx2;rotx3],B);
      bphi(Ix1)  = net.I(1:B^2/2);
      bphi(Ix2)  = net.I(B^2/2+1:B^2);
      end
      
      % rotate block back to original orientation
      bphi = sign(x3sum)*rot90(bphi,-rotation(idx));
      
      Snew = block_action(beta,lambda,bphi,np,B);
      DeltaS = Snew-Sold;
        
      % if action is lower we accept, if not we accept with prob:
      if rand<min(1,exp(-DeltaS))
          phi(Ip(bx:bx+B+1, by:by+B+1)) = bphi;
          Acc = Acc + 1;
          Delta = Delta + exp(-DeltaS);
      end
      end
      s(l) = action(beta,lambda,phi,np,nm);
      e(l) = nncorr(phi,np);
   end
   a = Acc/(Nprod*Netcycles)*100; 
   fprintf('Acceptance Rate was %0.3f %% \n',a);
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

function sb = block_action(beta,lambda,bphi,np,M)
    sb = 0;
    for lx=2:M+1
       sb = sb - beta*bphi(lx,1)*bphi(lx,2);
       sb = sb - beta*bphi(1,lx)*bphi(2,lx);
    for ly=2:M+1
       sb = sb + bphi(lx,ly)^2 + lambda*(bphi(lx,ly)^2-1)^2; 
       b = bphi(np(lx),ly)+bphi(lx,np(ly)); %positive nb interactions
       sb = sb - beta*b*bphi(lx,ly);
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
