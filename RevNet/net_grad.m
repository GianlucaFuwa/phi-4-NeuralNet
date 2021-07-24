% compute gradient of loss wrt weights for network with 1 hidden layer
function [dEdwF,dEdwG] = net_backpropBatch(net, goal)
dEdwF = cell(1,2);
dEdwG = cell(1,2);

% calculate gradients of cost wrt output and split in two halves
y_grad = net.dE(net.O,goal);
len = net.B^2/2;
y1_grad = y_grad(1:len); 
y2_grad = y_grad(len+1:end);

% calculate jacobian of G wrt y1
net.Gnet = net_dNetdI(net.Gnet,true);
net.Fnet = net_dNetdI(net.Fnet,false);
% calculate jacobian of each network's output wrt weights
net.Gnet = net_dNetdw(net.Gnet);
net.Fnet = net_dNetdw(net.Fnet);    
 
% calculate gradients of cost wrt weights
for l = 1:net.Fnet.Nlayers-1
dEdwF{l} = net.Fnet.dw{l} * (y1_grad + net.Gnet.dxT * y2_grad);
dEdwG{l} = net.Gnet.dw{l} * y2_grad;
end

end
