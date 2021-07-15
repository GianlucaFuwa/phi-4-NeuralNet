% Create and init RevNet
% args:
% Nneurons (2D vector, int) = Number of input and output neurons
% Mlayers (int) = Number of layers in ResBlocks
% Mneurons (MlayersD vector, int) = Number of neurons per layer in ResBlocks
% hasBiasNeuron (Mlayers-dim. vector, bool) = Bias neuron per layer in RBs
% B (int,even) = Block size
% beta, lambda (real) = Parameters of phi^4 simulation for calculation of action

function ffnet=make_ffnet(Nneurons, Mlayers, Mneurons, hasBiasNeuron, B, beta, lambda)
% only allow even number of output neurons in order for RevNet to function
if mod(Nneurons(1),2) ~= 0 || mod(Nneurons(2),2) ~= 0 
    error('Uneven number of neurons cant be split in half for RevNet')
end    

% create NN with 2 sub-networks F and G as resblocks
ffnet.Fnet.Nlayers       = Mlayers;
ffnet.Fnet.Nneurons      = Mneurons;
ffnet.Fnet.hasBiasNeuron = hasBiasNeuron;

ffnet.Gnet.Nlayers       = Mlayers;
ffnet.Gnet.Nneurons      = Mneurons;
ffnet.Gnet.hasBiasNeuron = hasBiasNeuron;

ffnet.Nlayers       = 2;
ffnet.Nneurons      = Nneurons;
ffnet.hasBiasNeuron = false;

ffnet.B = B;
ffnet.Beta = beta;
ffnet.Lambda = lambda;

% init input and output of network
ffnet.I = [];
ffnet.O = ones(ffnet.Nneurons(2),1);

% init activation and output of each layer except first, for all ResBlocks

for l=1:Mlayers
    if (1<l)&&(l<Mlayers) 
    ffnet.Fnet.O{l} = sparse(ones(ffnet.Fnet.Nneurons(l)+hasBiasNeuron(l),1));
    ffnet.Gnet.O{l} = sparse(ones(ffnet.Gnet.Nneurons(l)+hasBiasNeuron(l),1));
    elseif l == 1 || l == Mlayers
    ffnet.Fnet.O{l} = ones(ffnet.Fnet.Nneurons(l)+hasBiasNeuron(l),1);
    ffnet.Gnet.O{l} = ones(ffnet.Gnet.Nneurons(l)+hasBiasNeuron(l),1);    
    end
    if l>1
        ffnet.Fnet.I{l} = zeros(ffnet.Fnet.Nneurons(l),1);
        ffnet.Gnet.I{l} = zeros(ffnet.Gnet.Nneurons(l),1);
    else
        ffnet.Fnet.I{l} = [];
        ffnet.Gnet.I{l} = [];
    end
end

% set activation functions and their derivatives of each ResBlock
for l=2:Mlayers-1 
    ffnet.Fnet.actfunc{l}  = @(x) max(0,x);
    ffnet.Fnet.dactfunc{l} = @(x) double(x>0);
    ffnet.Gnet.actfunc{l}  = @(x) max(0,x);
    ffnet.Gnet.dactfunc{l} = @(x) double(x>0);
end

ffnet.Fnet.actfunc{Mlayers}  = @(x) x;
ffnet.Fnet.dactfunc{Mlayers} = @(x) ones('like',x);

ffnet.Gnet.actfunc{Mlayers}  = @(x) x;
ffnet.Gnet.dactfunc{Mlayers} = @(x) ones('like',x);

% init weights of ResBlocks as normally distributed numbers with set
% variance
for l=1:Mlayers-1 
    F_M = length(ffnet.Fnet.O{l});
    F_N = ffnet.Fnet.Nneurons(l+1);
    F_var = 2/(F_M+F_N); 
    G_M = length(ffnet.Fnet.O{l});
    G_N = ffnet.Gnet.Nneurons(l+1);
    G_var = 2/(G_M+G_N);
    ffnet.Fnet.w{l} = randn(F_M,F_N)*sqrt(F_var);
    ffnet.Gnet.w{l} = randn(G_M,G_N)*sqrt(G_var);
end

% init weight gradients for training
for l = 1:1:Mlayers-1
ffnet.Gnet.dwold{l} = zeros(numel(ffnet.Gnet.w{l}),1);
ffnet.Fnet.dwold{l} = zeros(numel(ffnet.Fnet.w{l}),1);
end

for l = 1:1:Mlayers-1
ffnet.Gnet.dw{l} = zeros(numel(ffnet.Gnet.w{l}),1);
ffnet.Fnet.dw{l} = zeros(numel(ffnet.Fnet.w{l}),1);
end

% define energy-function for training
ffnet.M  = sparse(couple1(B,1));
ffnet.S  = @(xout) sum(xout(1:B^2).^2) + sum(lambda*(xout(1:B^2).^2-1).^2) ...
           - beta*xout(1:B^2+4*B).'*ffnet.M*xout(1:B^2+4*B);
ffnet.dS = @(xout) 2*xout(1:B^2)+lambda*(4*xout(1:B^2).^3-4*xout(1:B^2)) ...
           - beta*(C(xout,ffnet.M,B));      
       
ffnet.E = @(xout,xtarget)  ffnet.S(xout)/(abs(ffnet.S(xtarget))+1e-6)/10+norm(xout(1:B^2)-xtarget(1:B^2))^2/B^2; 
ffnet.dE = @(xout,xtarget)  ffnet.dS(xout)/(abs(ffnet.S(xtarget))+1e-6)/10+2*(xout(1:B^2)-xtarget(1:B^2))/B^2;

% compute coupling matrix for action calculation 
function Mx = couple1(B,beta)
Ix1 = [];
Ix2 = [];
for lx=2:B+1
for ly=2:B+1
  n = (lx-1)*(B+2)+ly;
  if mod(lx+ly,2)==0
     Ix1 = [Ix1; n];
  else
     Ix2 = [Ix2; n];
  end
end
end
Ix3 = [2:B+1, (B+1)*(B+2)+2:(B+2)*(B+2)-1, ... % vertical boundaries
      B+3:B+2:B*(B+2)+1, 2*(B+2):B+2:(B+1)*(B+2)]'; % horizontal boundaries
Ip = zeros(B+2);
Ip(Ix1) = 1:length(Ix1);
Ip(Ix2) = length(Ix1)+1:length(Ix1)+length(Ix2);
Ip(Ix3) = length(Ix1)+length(Ix2)+1:length(Ix1)+length(Ix2)+length(Ix3);
np = [2:B+2,1];    % np(l) is the positive neighbor of l
nm = [B+2, 1:B+2-1]; %nm(l) is the negative neighbor of l

Mx = zeros(length(Ix1)+length(Ix2)+length(Ix3));
for i = 1:length(Ix1)+length(Ix2)
[row,col] = ind2sub([B+2,B+2],find(Ip==i));
neighx = Ip(row,np(col));
neighy = Ip(np(row),col);
Mx(neighx,i) = beta;
Mx(neighy,i) = beta;
end 
for j = length(Ix1)+length(Ix2)+1:length(Ix1)+length(Ix2)+B
[row,col] = ind2sub([B+2,B+2],find(Ip==j));
neighx = Ip(row,np(col));
Mx(neighx,j) = beta;
end 
for k = length(Ix1)+length(Ix2)+2*B+1:length(Ix1)+length(Ix2)+3*B
[row,col] = ind2sub([B+2,B+2],find(Ip==k));
neighy = Ip(np(row),col);
Mx(neighy,k) = beta;
end 
end

function part = C(xout,M,B)
result =  (M.'+M)*xout(1:B^2+4*B);
part = result(1:B^2);
end

end
