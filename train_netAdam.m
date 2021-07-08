L = 12;
M = 8;
beta = 0.6586246;
%beta = 0.61567;
lambda = 0.25;
%lambda = 0.5;
Nprod = 50000;

Adam8_400rotSdist = make_ffnet([98 98],3,[66 400 32],[true true false],M,beta,lambda);
%Adam8_400Wrot = Adam8_400Wrot;
Ntrain = 10000;
Nvalid = 5000;
Epochs = 100;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
alpha = 1e-4;
regu = 0;
eta = 1;

trainx = trainsetrot{1}(1:end,1:Ntrain);
trainy = trainsetrot{2}(1:end,1:Ntrain);
validx = trainsetrot{1}(1:end,Ntrain+1:Ntrain+Nvalid);
validy = trainsetrot{2}(1:end,Ntrain+1:Ntrain+Nvalid);

meanEtrain = zeros(1,Epochs+1);
meanEvalid = zeros(1,Epochs+1);
accrates = zeros(1,Epochs+1);
taues = zeros(1,Epochs+1);
tauss = zeros(1,Epochs+1);
GoodTaus = cell(1,Epochs+1);

numweights = Adam8_400rotSdist.Fnet.Nlayers-1;
mF = cell(1,numweights);
vF = cell(1,numweights);
mG = cell(1,numweights);
vG = cell(1,numweights);
for l = 1:numweights
mF{l} = zeros(numel(Adam8_400rotSdist.Fnet.w{l}),1);
vF{l} = zeros(numel(Adam8_400rotSdist.Fnet.w{l}),1);
mG{l} = zeros(numel(Adam8_400rotSdist.Gnet.w{l}),1);
vG{l} = zeros(numel(Adam8_400rotSdist.Gnet.w{l}),1);
end

shuffle = @(v)v(randperm(numel(v)));
t = 0;
it =  Ntrain*Epochs;
for i = 1:Epochs+1
    Etottrain = 0;
    Etotval = 0;
    if i == 1
        for l = shuffle(1:Ntrain)
            Adam8_400rotSdist = ffnet_forward(Adam8_400rotSdist,trainx(:,l),M);
            E = Adam8_400rotSdist.E(Adam8_400rotSdist.O,trainy(1:end,l));
            Etottrain = Etottrain + E;
        end
    else     
    for l = shuffle(1:Ntrain)
    t = t+1;  
    Adam8_400rotSdist = ffnet_forward(Adam8_400rotSdist,trainx(:,l),M);
    [gF,gG] = ffnet_backpropBatch(Adam8_400rotSdist,trainy(:,l));
    alphat = alpha*sqrt(1-beta2^t)/(1-beta1^t);    
    for a = 1:numweights
        vF{a} = beta2*vF{a}+(1-beta2)*gF{a}.^2;
        vG{a} = beta2*vG{a}+(1-beta2)*gG{a}.^2;
        % Demon        
        betat = beta1*(1-t/(it-Epochs/50*Ntrain))/((1-beta1)+beta1*(1-t/(it-Epochs/50*Ntrain))); 
        mF{a} = betat*mF{a}+gF{a};
        mG{a} = betat*mG{a}+gG{a};
        %
        Adam8_400rotSdist.Fnet.w{a}(:) = Adam8_400rotSdist.Fnet.w{a}(:) ...
                - eta*(alphat*(mF{a}./(sqrt(vF{a})+epsilon))+regu*Adam8_400rotSdist.Fnet.w{a}(:));
        Adam8_400rotSdist.Gnet.w{a}(:) = Adam8_400rotSdist.Gnet.w{a}(:) ...
                - eta*(alphat*(mG{a}./(sqrt(vG{a})+epsilon))+regu*Adam8_400rotSdist.Gnet.w{a}(:));
    end   
    E = Adam8_400rotSdist.E(Adam8_400rotSdist.O,trainy(1:end,l));
    Etottrain = Etottrain + E;
    end
    end

    for j = 1:Nvalid
        Adam8_400rotSdist = ffnet_forward(Adam8_400rotSdist,validx(1:end,j),M);
        Evalid = Adam8_400rotSdist.E(Adam8_400rotSdist.O,validy(1:end,j));    
        Etotval = Etotval + Evalid;
    end

    meanEtrain(i) = Etottrain/Ntrain;
    meanEvalid(i) = Etotval/Nvalid;
    if i == 1
        fprintf('Mean E of assessment cycle was: %0.10f\n',Etottrain/Ntrain);
    else 
    fprintf('Mean E of training cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etottrain/Ntrain,Etottrain/Ntrain-meanEtrain(i-1));
    fprintf('Mean E of validation cycle %4d was: %0.10f (%+0.10f)\n',i-1,Etotval/Nvalid,Etotval/Nvalid-meanEvalid(i-1));
    end
    [e,s,a] = phi4MC_rotexam(L,M,beta,lambda,Adam8_400rotSdist,Nprod,1);
    [~,~,~,taue,~]=UWerr(e,1.5,length(e),1);
    [~,~,~,taus,~]=UWerr(s,1.5,length(s),1);
    accrates(i) = a/100; 
    taues(i) = taue*a/100;
    tauss(i) = taus*a/100;
    fprintf('taue was: %0.10f\n',taues(i));
    fprintf('taus was: %0.10f\n',tauss(i));
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
save('Adam8_400rotSdist.mat','Adam8_400rotSdist','meanEtrain','meanEvalid','accrates','taues','tauss')

figure
plot(1:Epochs+1, meanEtrain/meanEtrain(1), 'bo-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(1:Epochs+1, meanEvalid/meanEtrain(1), 'ro-', 'LineWidth', 1, 'MarkerSize', 3);
hold on
plot(1:Epochs+1, accrates, 'ko-', 'Linewidth', 1, 'MarkerSize', 3);
hold on
plot(1:Epochs+1, taues/max(taues), 'mo-', 'Linewidth', 1, 'MarkerSize', 3);
hold off

title('Training Evaluation h =' );
xlabel('Epoch Number');
ylabel('Avg Loss');
xlim([0 Epochs+1]);
legend({'Training','Validation','Acc. Rate'},'Location','southwest');


clear a alpha alphat beta1 beta2 betat E Epochs epsilon Etottrain Etotval Evalid...
    gF gG i j l mF mG Ntrain Nvalid shuffle t trainx trainy validx validy...
    vF vG e s Nprod M beta Eit it eta L numweights regu lambda