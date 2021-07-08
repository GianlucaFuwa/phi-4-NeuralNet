function net=net_eval(net, input_layer)

net.O{1}(1:net.Nneurons(1)) = input_layer;

for l = 2:net.Nlayers
   net.I{l}(1:net.Nneurons(l)) = net.O{l-1}.' * net.w{l-1};
   net.O{l}(1:net.Nneurons(l)) = net.actfunc{l}(net.I{l});
end 
 
