function [dsTrain,layers,options] = train_lstm_model(sampleFile, trainParams)
% train a LSTM-based model
    ds = load(sampleFile);
    numSamples = trainParams.numSamples;    

    % preprocess data for training
    % Refer to the Help "Import Data into Deep Network Designer / Sequences and time series" 
    initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
    states = {};
    times = [];
    labels = [];
    for i=1:numSamples
        data = load(ds.samples{i,1}).state;
        t = data(1,:);
        x = data(2:7,:);
        for tInit = initTimes
            initIdx = find(t >= tInit, 1, 'first');
            startIdx = initIdx-trainParams.sequenceStep+1;
            t0 = t(initIdx);
            x0 = [t(startIdx:initIdx); x(:,startIdx:initIdx)];
            for j=initIdx+1:length(t)
                states{end+1} = x0;
                times = [times, t(j)-t0];
                labels = [labels, x(:,j)];
            end
        end
    end
    % disp(num2str(length(times)) + " samples are generated for training.");
    states = reshape(states, [], 1);
    times = times';
    labels = labels';

    % combine a datastore for training
    miniBatchSize = trainParams.miniBatchSize;
    dsState = arrayDatastore(states,'OutputType',"same",'ReadSize',miniBatchSize);
    dsTime = arrayDatastore(times,'ReadSize',miniBatchSize);
    dsLabel = arrayDatastore(labels,'ReadSize',miniBatchSize);
    dsTrain = combine(dsState, dsTime, dsLabel);

    % Create neural network
    numStates = 6;
    layers = [
        sequenceInputLayer(numStates+1)
        lstmLayer(trainParams.HiddenState, OutputMode = "last")
        concatenationLayer(1, 2, Name = "cat")
        ];
    
    numMiddle = floor(trainParams.numLayers/2);
    for i = 1:numMiddle
        layers = [
            layers
            fullyConnectedLayer(trainParams.numNeurons)
            tanhLayer
        ];
    end
    layers = [
        layers
        dropoutLayer(trainParams.dropoutFactor)
        ];
    for i = numMiddle+1:trainParams.numLayers
        layers = [
            layers
            fullyConnectedLayer(trainParams.numNeurons)
            tanhLayer
        ];
    end
    
    layers = [
        layers
        fullyConnectedLayer(numStates, "Name", "output")
        weightedLossLayer("mse")
       ];
    
    layers = layerGraph(layers);
    layers = addLayers(layers,[...
        featureInputLayer(1, Name = "time")]);
    layers = connectLayers(layers, "time", "cat/in2");
    % plot(lgraph);

    options = trainingOptions("adam", ...
        InitialLearnRate = trainParams.learningRate, ...
        MaxEpochs = trainParams.numEpochs, ...
        MiniBatchSize = miniBatchSize, ...
        SequencePaddingDirection = "left", ...
        Shuffle = "every-epoch", ...
        Plots = "training-progress", ...
        Verbose = false, ...
        LearnRateSchedule = "piecewise", ...
        LearnRateDropFactor = trainParams.LearnRateDropFactor, ...
        LearnRateDropPeriod = floor(trainParams.numEpochs/3));
