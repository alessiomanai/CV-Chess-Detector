
% Deep Learning: Transfer Learning in 10 Lines of MATLAB Code 

% Load Training Images
imds = imageDatastore('pezzinew', 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

%tbl = countEachLabel(imds);

%minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
%imds = splitEachLabel(imds, minSetCount, 'randomize');

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
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,...
    'MaxEpochs', 20, 'MiniBatchSize', 64);

% Set custom read function 
% One of the great things about imageDataStore it lets you specify a
% "custom" read function, in this case it is simply resizing the input
% images to 227x227 pixels which is what AlexNet expects. You can do this by
% specifying a function handle of a function with code to read and
% pre-process the image. 

trainingImages.ReadFcn = @readFunctionTrain;

% Train the Network 
% This process usually takes about 5-20 minutes on a desktop GPU. 
myNet = trainNetwork(trainingImages, layers, opts);


% Test Network Performance
% Now let's the test the performance of our new "snack recognizer" on the test set.
testImages.ReadFcn = @readFunctionTrain;
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

