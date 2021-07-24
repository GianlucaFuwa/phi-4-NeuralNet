% set simulation parameters for network and check-ups conducted during training
L = 12;
M = 8;
beta = 0.6586246;
lambda = 0.25;
Nprod = 100000;

Net = make_ffnet([98 98],3,[66 400 32],[true true false],M,beta,lambda);

Ntrain = 10000;
Nvalid = 5000;
Epochs = 100;
beta1 = 0.9; % init momentum
beta2 = 0.999;
epsilon = 1e-8;
alpha = 1e-4; % init learining rate
regu = 5e-5; % weight decay parameter
eta = 1;

% init training set
trainx = trainsetrot{1}(1:end,1:Ntrain);
trainy = trainsetrot{2}(1:end,1:Ntrain);
validx = trainsetrot{1}(1:end,Ntrain+1:Ntrain+Nvalid);
validy = trainsetrot{2}(1:end,Ntrain+1:Ntrain+Nvalid);

meanEtrain = zeros(1,Epochs+1);
meanEvalid = zeros(1,Epochs+1);
accrates = zeros(1,Epochs+1);
taues = zeros(1,Epochs+1);
tauss = zeros(1,Epochs+1);

numweights = Net.Fnet.Nlayers-1;
mF = cell(1,numweights);
vF = cell(1,numweights);
mG = cell(1,numweights);
vG = cell(1,numweights);
for l = 1:numweights
mF{l} = zeros(numel(Net.Fnet.w{l}),1);
vF{l} = zeros(numel(Net.Fnet.w{l}),1);
mG{l} = zeros(numel(Net.Gnet.w{l}),1);
vG{l} = zeros(numel(Net.Gnet.w{l}),1);
end

shuffle = @(v)v(randperm(numel(v)));
t = 0;
it =  Ntrain*Epochs;
for i = 1:Epochs+1
    Etottrain = 0;
    Etotval = 0;
    if i == 1
        for l = shuffle(1:Ntrain)
            Net = ffnet_forward(Net,trainx(:,l),M);
            E = Net.E(Net.O,trainy(1:end,l));
            Etottrain = Etottrain + E;
        end
    else     
    for l = shuffle(1:Ntrain) % shuffle trainingset in each epoch
    t = t+1;  
    Net = ffnet_forward(Net,trainx(:,l),M);
    [gF,gG] = ffnet_backpropBatch(Net,trainy(:,l)); % compute gradients
    alphat = alpha*sqrt(1-beta2^t)/(1-beta1^t);    
    for a = 1:numweights
        vF{a} = beta2*vF{a}+(1-beta2)*gF{a}.^2;
        vG{a} = beta2*vG{a}+(1-beta2)*gG{a}.^2;
        % Decaying momentum        
        betat = beta1*(1-t/(it-Epochs/50*Ntrain))/((1-beta1)+beta1*(1-t/(it-Epochs/50*Ntrain))); 
        mF{a} = betat*mF{a}+gF{a};
        mG{a} = betat*mG{a}+gG{a};
        % Update weights
        Net.Fnet.w{a}(:) = Net.Fnet.w{a}(:) ...
                - eta*(alphat*(mF{a}./(sqrt(vF{a})+epsilon))+regu*Net.Fnet.w{a}(:));
        Net.Gnet.w{a}(:) = Net.Gnet.w{a}(:) ...
                - eta*(alphat*(mG{a}./(sqrt(vG{a})+epsilon))+regu*Net.Gnet.w{a}(:));
    end   
    E = Net.E(Net.O,trainy(1:end,l));
    Etottrain = Etottrain + E;
    end
    end

    for j = 1:Nvalid
        Net = ffnet_forward(Net,validx(1:end,j),M);
        Evalid = Net.E(Net.O,validy(1:end,j));    
        Etotval = Etotval + Evalid;
    end

    meanEtrain(i) = Etottrain/Ntrain;
    meanEvalid(i) = Etotval/Nvalid;
    % track loss
    if i == 1
        fprintf('Mean E of assessment cycle was: %0.10f\n',Etottrain/Ntrain);
    else 
    fprintf('Mean E of training cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etottrain/Ntrain,Etottrain/Ntrain-meanEtrain(i-1));
    fprintf('Mean E of validation cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etotval/Nvalid,Etotval/Nvalid-meanEvalid(i-1));
    end
    [~,~,~,~,a] = phi4Net_rot(L,M,beta,lambda,Net,Nprod,1,0);
    accrates(i) = a/100; % track acceptance rate
    if mod(i,Epochs/10) == 0
        eta = eta/2;
    end
end
save('Net.mat','Net','meanEtrain','meanEvalid','accrates')

figure
plot(1:Epochs+1, meanEtrain/meanEtrain(1), 'bo-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(1:Epochs+1, meanEvalid/meanEtrain(1), 'ro-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(1:Epochs+1, accrates, 'ko-', 'Linewidth', 1, 'MarkerSize', 3);
hold off

title('Training Evaluation');
xlabel('Epoch Number');
ylabel('Avg Loss');
xlim([0 Epochs+1]);
legend({'Training','Validation','Acc. Rate'},'Location','west');


clear a alpha alphat beta1 beta2 betat E Epochs epsilon Etottrain Etotval Evalid...
    gF gG i j l mF mG Ntrain Nvalid shuffle t trainx trainy validx validy...
    vF vG e s Nprod M beta Eit it eta L numweights regu lambda
