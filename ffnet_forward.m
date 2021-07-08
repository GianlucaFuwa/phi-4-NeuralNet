% Regular operation mode of neural network with input_layer given 
% in phi4MCNN.m

function ffnet = ffnet_forward(ffnet, input_layer, M)
%x1_length = ffnet.Nneurons(2)/2;
x1_length = M^2/2;

x1 = input_layer(1:x1_length); % slice input into 3 vectors x1, x2, x3
x2 = input_layer(x1_length+1:2*x1_length);
x3 = input_layer(2*x1_length+1:ffnet.Nneurons(1));
% x3 contains neighbours and beta, lambda

% set input as first layer of NN and evaluate as ordered for RevNet
ffnet.I = [x1;x2;x3];

ffnet.Fnet = ffnet_eval(ffnet.Fnet, [x2;x3]);
y1 = x1 + ffnet.Fnet.O{ffnet.Fnet.Nlayers};

ffnet.Gnet = ffnet_eval(ffnet.Gnet, [y1;x3]);
y2 = x2 + ffnet.Gnet.O{ffnet.Gnet.Nlayers};

% set ouput of last layer
ffnet.O = [y1;y2;x3];
end





