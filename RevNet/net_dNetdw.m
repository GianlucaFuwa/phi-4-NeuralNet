% compute gradients of 1-hidden-layer residual block output wrt weights
% ReLu activation function in all layers except for first and last required
function net = net_dNetdw(net)

C2 = cell(1,length(net.I{3}));
C2(:) = {sparse(net.O{2})};
net.dw{2} = blkdiag(C2{:});

C1 = cell(1,length(net.I{2}));
C1(:) = {sparse(net.O{1})};
net.dw{1} = blkdiag(C1{:}) * net.dO2dI2T * gpuArray(net.w{2});    

end
