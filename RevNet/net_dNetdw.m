% compute gradients of 1-hidden-layer residual block output wrt weights
% identity activation required in first and last layer
function net = net_dNetdw(net)

C2 = cell(1,length(net.I{3}));
C2(:) = {sparse(net.O{2})}; % compute dI3dw2
net.dw{2} = blkdiag(C2{:});

C1 = cell(1,length(net.I{2}));
C1(:) = {sparse(net.O{1})}; % compute dI2dw1
net.dw{1} = blkdiag(C1{:}) * net.dO2dI2T * gpuArray(net.w{2});    

end
