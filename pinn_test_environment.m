%% Description
% Test environment for trained models

close all;
clear; 
clc;

%% settings
sysParams = params_system();
trainParams = params_training();
ctrlParams = params_control();
tSpan = 0:0.002:10;
tRMSE = floor(length(tSpan)/2); % time steps not in rmse calculation
tForceStop = 1;
trainParams.type = "pinn";
predInterval = 10;

modelFile = "best_pinn_models.mat";

F1Min = max(5,sysParams.fc_max);
Fmax = 10;

%% Test 1
net = load(modelFile).model_6_256_400.trainedNet;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
xp = predict_motion(net, trainParams.type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));
initIdx = find(t >= tForceStop,1,'first');
% x0 = x(initIdx, :);
% t0 = t(initIdx);
% xp = zeros(length(tp),6);
% for i = 1 : length(tp)
%     xInit = dlarray([x0, tp(i)-t0]', 'CB');
%     xp(i,:) = extractdata(predict(net, xInit));
% end

rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 1", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,t(initIdx+1:end),xp(initIdx+1:end,:),titletext)
avgErr = evaluate_model(net, sysParams, ctrlParams, trainParams);

%% root square error function
function rse = root_square_err(indices, x, xp)
    % root square error of prediction and reference
    numPoints = length(indices);
    x_size = size(xp);
    errs = zeros(x_size(2), numPoints);
    for i = 1 : numPoints
        for j = 1:x_size(2)
            errs(j, i) = x(indices(i), j) - xp(indices(i), j);
        end
    end
    rse = sqrt(errs.^2);
end