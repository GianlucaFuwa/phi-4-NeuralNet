% Reverse operation mode of neural network with input_layer given 
% in phi4MCNN.m

function ffnet = ffnet_backward(ffnet, output_layer, M)
%y1_length = ffnet.Nneurons(2)/2;
y1_length = M^2/2;

y1 = output_layer(1:y1_length);  
y2 = output_layer(y1_length+1:2*y1_length);
x3 = output_layer(2*y1_length+1:ffnet.Nneurons(1)); 

ffnet.O = [y1;y2;x3];

ffnet.Gnet = ffnet_eval(ffnet.Gnet, [y1;x3]);
x2 = y2 - ffnet.Gnet.O{ffnet.Gnet.Nlayers};

ffnet.Fnet = ffnet_eval(ffnet.Fnet, [x2;x3]);
x1 = y1 - ffnet.Fnet.O{ffnet.Fnet.Nlayers};

ffnet.I = [x1;x2;x3];
end