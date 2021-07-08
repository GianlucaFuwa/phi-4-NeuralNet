function ffnet=ffnet_eval(ffnet, input_layer)

ffnet.O{1}(1:ffnet.Nneurons(1)) = input_layer;

for l = 2:ffnet.Nlayers-1
   ffnet.I{l}(1:ffnet.Nneurons(l)) = ffnet.O{l-1}.' * ffnet.w{l-1};
   ffnet.O{l}(1:ffnet.Nneurons(l)) = max(0,ffnet.I{l});
end 
ffnet.I{ffnet.Nlayers}(1:ffnet.Nneurons(ffnet.Nlayers)) = ffnet.O{ffnet.Nlayers-1}.' * ffnet.w{ffnet.Nlayers-1};
ffnet.O{ffnet.Nlayers}(1:ffnet.Nneurons(ffnet.Nlayers)) = ffnet.I{ffnet.Nlayers};
 