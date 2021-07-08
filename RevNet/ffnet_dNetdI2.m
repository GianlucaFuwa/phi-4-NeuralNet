function net = ffnet_dNetdI2(net,G)

leny1 = length(net.O{4});
lenO1 = length(net.O{1});

net.dO1dy1T = speye(leny1,lenO1);

lenI2 = length(net.I{2});
lenO2 = length(net.O{2});

net.dO2dI2T = speye(lenI2,lenO2);
net.dO2dI2T(1:lenI2+1:end) = net.dactfunc{2}(net.I{2});

lenI3 = length(net.I{3});
lenO3 = length(net.O{3});

net.dO3dI3T = speye(lenI3,lenO3);
net.dO3dI3T(1:lenI3+1:end) = net.dactfunc{3}(net.I{3});

if G == true
net.dxT = (net.dO1dy1T * net.w{1}) * (net.dO2dI2T * net.w{2}) ...
          * (net.dO3dI3T * net.w{3});
end
end