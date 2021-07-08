% Set Parameters for phi^4 simulation
L = 12;
M = 8;
beta = 0.61567;
lambda = 0.5;
Nprod = 5000; 

% Init Neural Network
SGDM8_400rot2 = make_ffnet([98 98],3,[66 400 32],[true true false],M,beta,lambda);
%SGDM8_400W = Net;

% set training options
Ntrain = 10000;
Nvalid = 5000;
Epochs = 500;
alpha = 0.1;
rho = 0.01; %->0.005
regu = 1e-5; 
%delay = 0;

numweights = SGDM8_400rot2.Fnet.Nlayers-1;

for l = 1:numweights
SGDM8_400rot2.Gnet.dwold{l} = zeros(numel(SGDM8_400rot2.Gnet.w{l}),1);
SGDM8_400rot2.Fnet.dwold{l} = zeros(numel(SGDM8_400rot2.Fnet.w{l}),1);
end

% Get Training-Set
trainx = trainsetrot{1}(1:end,Nvalid+1:Nvalid+Ntrain);
trainy = trainsetrot{2}(1:end,Nvalid+1:Nvalid+Ntrain);
validx = trainsetrot{1}(1:end,1:Nvalid);
validy = trainsetrot{2}(1:end,1:Nvalid);

% Init for Plot
numep = zeros(1,Epochs+1);
meanEtrain = zeros(1,Epochs+1);

meanEvalid = zeros(1,Epochs+1);
accrates = zeros(1,Epochs+1);
shuffle = @(v)v(randperm(numel(v)));
t = 0;
eta = 1;
it = Epochs*Ntrain;
for i = 1:Epochs+1
    Etottrain = 0;
    Etotval = 0;
    % Assessment cycle
    if i == 1
        for l = shuffle(1:Ntrain)
            SGDM8_400rot2 = ffnet_forward(SGDM8_400rot2,trainx(:,l),M);
            E = SGDM8_400rot2.E(SGDM8_400rot2.O,trainy(:,l));
            Etottrain = Etottrain + E;
        end
    else 
    for l = shuffle(1:Ntrain) 
    t = t+1;
    % DEMON momentum
    alphat = alpha*(1-t/(it-Epochs/50*Ntrain))/((1-alpha)+alpha*(1-t/(it-Epochs/50*Ntrain)));
    % Evaluate NN and compute Gradients
    SGDM8_400rot2 = ffnet_forward(SGDM8_400rot2,trainx(:,l),M);
    [gF,gG] = ffnet_backpropBatch(SGDM8_400rot2,trainy(:,l));
    % Change weights according to SGDM with L1-Regularization
    for a = 1:numweights    
    SGDM8_400rot2.Gnet.dwold{a} = -eta*rho*gG{a} + alphat*SGDM8_400rot2.Gnet.dwold{a};
    SGDM8_400rot2.Gnet.w{a}(:)  = SGDM8_400rot2.Gnet.w{a}(:) + SGDM8_400rot2.Gnet.dwold{a}...
                                       -eta*regu*SGDM8_400rot2.Gnet.w{a}(:);
    SGDM8_400rot2.Fnet.dwold{a} = -eta*rho*gF{a} + alphat*SGDM8_400rot2.Fnet.dwold{a};
    SGDM8_400rot2.Fnet.w{a}(:)  = SGDM8_400rot2.Fnet.w{a}(:) + SGDM8_400rot2.Fnet.dwold{a}...
                                       -eta*regu*SGDM8_400rot2.Fnet.w{a}(:);
    end  
    % Calculate Energy
    E = SGDM8_400rot2.E(SGDM8_400rot2.O,trainy(:,l));
    Etottrain = Etottrain + E;
    end
    end
% Test NN on validation cycle
for j = 1:Nvalid
    SGDM8_400rot2 = ffnet_forward(SGDM8_400rot2,validx(1:end,j),M);
    Evalid = SGDM8_400rot2.E(SGDM8_400rot2.O,validy(1:end,j));    
    Etotval = Etotval + Evalid;
end
    numep(i) = i;
    meanEtrain(i) = Etottrain/Ntrain;
    meanEvalid(i) = Etotval/Nvalid;
    if i == 1
        fprintf('Mean E of assessment cycle was: %0.10f\n',Etottrain/Ntrain);
    else
    fprintf('Mean E of training cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etottrain/Ntrain,Etottrain/Ntrain-meanEtrain(i-1));
    fprintf('Mean E of validation cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etotval/Nvalid,Etotval/Nvalid-meanEvalid(i-1));
    end
    % Test NN acceptance rate on phi^4 Simulation
    [~,~,a] = phi4MC_rotexam(L,M,beta,lambda,SGDM8_400rot2,Nprod,1);
    accrates(i)=a/100;
    % Reduce Values of max and min learning rate
    %{
    if i == Epochs*0.5
        eta = eta*0.1;
    elseif i == round(Epochs*0.75)
        eta = eta*0.1;
    elseif i == round(Epochs*0.875)
        eta = eta*0.1;
    end
    %}
    if mod(i,Epochs/10) == 0
        eta = eta/2;
    end
end

% Plot training and validation losses and acceptance rates
figure
plot(numep, meanEtrain/meanEtrain(1), 'bo-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(numep, meanEvalid/meanEtrain(1), 'ro-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(numep, accrates, 'ko-', 'Linewidth', 1, 'MarkerSize', 3);
hold off

title('Training Evaluation h =' );
xlabel('Epoch Number');
ylabel('Avg Loss');
xlim([-3 Epochs+10]);
legend({'Training','Validation','Acc. Rate'},'Location','southwest');

clear Nprod beta lambda a alpha rho E Epochs Etottrain Etotval Evalid...
    i j l Ntrain Nvalid shuffle t trainx trainy validx validy...
    L M numweights regu gF gG delay alphat it CosAn eta