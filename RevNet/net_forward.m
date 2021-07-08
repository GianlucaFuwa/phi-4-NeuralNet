% Regular operation mode of neural network

function net = net_forward(net, input_layer, M)
x1_length = M^2/2;

x1 = input_layer(1:x1_length); % slice input into 3 vectors x1, x2, x3
x2 = input_layer(x1_length+1:2*x1_length);
x3 = input_layer(2*x1_length+1:net.Nneurons(1));
% x3 contains neighbours and beta+lambda

% set input as first layer of NN and evaluate as ordered for RevNet
net.I = [x1;x2;x3];

net.Fnet = net_eval(net.Fnet, [x2;x3]);
y1 = x1 + net.Fnet.O{net.Fnet.Nlayers};

net.Gnet = net_eval(net.Gnet, [y1;x3]);
y2 = x2 + net.Gnet.O{net.Gnet.Nlayers};

% set ouput of last layer
net.O = [y1;y2;x3];
end





