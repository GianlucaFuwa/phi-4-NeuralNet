% calc gradient of network output wrt weights for 2-hidden-layer network
% identity activation required in first and last layer
function net = net_dNetdw2(net)

dO3dI3Tw3 = net.dO3dI3T * gpuArray(net.w{3});

% Gradient wrt 3rd weight matrix
C3 = cell(1,length(net.I{4}));
C3(:) = {sparse(net.O{3})};
net.dw{3} = blkdiag2(C3{:});

% Gradient wrt 2nd weight matrix
C2 = cell(1,length(net.I{3}));
C2(:) = net.O(2);
net.dw{2} = blkdiag2(C2{:}) * dO3dI3Tw3;    

% Gradient wrt 1st weight matrix
C1 = cell(1,length(net.I{2}));
C1(:) = {sparse(net.O{1})};
net.dw{1} = blkdiag2(C1{:}) * net.dO2dI2T * (gpuArray(net.w{2}) * dO3dI3Tw3);      

end
