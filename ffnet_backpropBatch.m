% compute gradient of loss wrt weights
function [dEdwF,dEdwG] = ffnet_backpropBatch(net, goal)
dEdwF = cell(1,2);
dEdwG = cell(1,2);

% calculate gradients of cost wrt output and split in two halves
y_grad = net.dE(net.O,goal);
len = net.B^2/2;
y1_grad = y_grad(1:len); 
y2_grad = y_grad(len+1:end);

% determine number of weight matrices and compute Gradients of Net. Output
if net.Gnet.Nlayers == 3
    % calculate jacobian of G wrt y1
    net.Gnet = ffnet_dNetdI(net.Gnet,true);
    net.Fnet = ffnet_dNetdI(net.Fnet,false);
    % calculate jacobian of each network's output wrt weights
    net.Gnet = ffnet_dNetdw(net.Gnet);
    net.Fnet = ffnet_dNetdw(net.Fnet);    
elseif net.Gnet.Nlayers == 4
    % calculate jacobian of G wrt y1
    net.Gnet = ffnet_dNetdI2(net.Gnet,true);
    net.Fnet = ffnet_dNetdI2(net.Fnet,false);
    % calculate jacobian of each network's output wrt weights
    net.Gnet = ffnet_dNetdw2(net.Gnet);
    net.Fnet = ffnet_dNetdw2(net.Fnet); 
elseif net.Gnet.Nlayers == 5
    % calculate jacobian of G wrt y1
    net.Gnet = ffnet_dNetdI3(net.Gnet,true);
    net.Fnet = ffnet_dNetdI3(net.Fnet,false);
    % calculate jacobian of each network's output wrt weights
    net.Gnet = ffnet_dNetdw3(net.Gnet);
    net.Fnet = ffnet_dNetdw3(net.Fnet);    
end
 
% calculate gradients of cost wrt weights
for l = 1:net.Fnet.Nlayers-1
dEdwF{l} = net.Fnet.dw{l} * (y1_grad + net.Gnet.dxT * y2_grad);
dEdwG{l} = net.Gnet.dw{l} * y2_grad;
end

end