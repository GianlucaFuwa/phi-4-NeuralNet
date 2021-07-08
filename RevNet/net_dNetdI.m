% compute gradients of 1-hidden-layer residual block output wrt input
% identity activation in required first and last layer
function net = net_dNetdI(net,G)

leny1 = length(net.O{3});
lenO1 = length(net.O{1});

net.dO1dy1T = speye(leny1,lenO1);

lenI2 = length(net.I{2});
lenO2 = length(net.O{2});

net.dO2dI2T = sparse(lenI2,lenO2);
net.dO2dI2T(1:lenI2+1:end) = net.dactfunc{2}(net.I{2});

% calculate gradient only if the network is res block G
if G == true
net.dxT = net.dO1dy1T * net.w{1} * net.dO2dI2T * net.w{2};
end
end
