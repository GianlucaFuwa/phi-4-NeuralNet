% Reverse operation mode of neural network

function net = net_backward(net, output_layer, M)
y1_length = M^2/2;

y1 = output_layer(1:y1_length);  
y2 = output_layer(y1_length+1:2*y1_length);
x3 = output_layer(2*y1_length+1:net.Nneurons(1)); 

net.O = [y1;y2;x3];

net.Gnet = net_eval(net.Gnet, [y1;x3]);
x2 = y2 - net.Gnet.O{net.Gnet.Nlayers};

net.Fnet = net_eval(net.Fnet, [x2;x3]);
x1 = y1 - net.Fnet.O{net.Fnet.Nlayers};

net.I = [x1;x2;x3];
end
