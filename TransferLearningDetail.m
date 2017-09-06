
imds = imageDatastore('pezzinew', 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(imds, 0.8, 'randomize');
 
% Load Pre-trained Network (AlexNet)
alex = alexnet; 

% Review Network Architecture 
layers = alex.Layers 

% Modify Pre-trained Network 
layers(23) = fullyConnectedLayer(9); % change this based on # of classes
layers(25) = classificationLayer

% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly.
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,...
    'MaxEpochs', 20, 'MiniBatchSize', 64);

% Set custom read function 
trainingImages.ReadFcn = @readFunctionTrain;

% Train the Network 
% This process usually takes about 5-20 minutes on a desktop GPU. 
myNet = trainNetwork(trainingImages, layers, opts);


% Test Network Performance
testImages.ReadFcn = @readFunctionTrain;
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

