
function [net, featureLayer, classifier] = DeepLearningImageClassification

    setDir  = fullfile('pezzinew');
    imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

    tbl = countEachLabel(imds)

    % Because |imds| above contains an unequal number of images per category,
    % let's first adjust it, so that the number of images in the training set
    % is balanced.

    minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

    % Use splitEachLabel method to trim the set.
    imds = splitEachLabel(imds, minSetCount, 'randomize');

    % Notice that each set now has exactly the same number of images.
    countEachLabel(imds)

    % Load pre-trained AlexNet
    net = alexnet()

    % View the CNN architecture
    net.Layers

    % The first layer defines the input dimensions. Each CNN has a different
    % input size requirements. The one used in this example requires image
    % input that is 227-by-227-by-3.

    % Inspect the first layer
    net.Layers(1)

    % Inspect the last layer
    net.Layers(end)

    % Number of class names for ImageNet classification task
    numel(net.Layers(end).ClassNames)

    % Pre-process Images For CNN

    % Set the ImageDatastore ReadFcn
    imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

        function Iout = readAndPreprocessImage(filename)

            I = imread(filename);

            % Some images may be grayscale. Replicate the image 3 times to
            % create an RGB image. 
            if ismatrix(I)
                I = cat(3,I,I,I);
            end

            % Resize the image as required for the CNN. 
            Iout = imresize(I, [227 227]);  
        end

    % Prepare Training and Test Image Sets
    [trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

    % Extract Training Features Using CNN
    % Get the network weights for the second convolutional layer
    w1 = net.Layers(2).Weights;

    % Scale and resize the weights for visualization
    w1 = mat2gray(w1);
    w1 = imresize(w1,5); 

    featureLayer = 'fc7';
    trainingFeatures = activations(net, trainingSet, featureLayer, ...
        'MiniBatchSize', 32, 'OutputAs', 'columns');

    % Train A Multiclass SVM Classifier Using CNN Features
    % Get training labels from the trainingSet
    trainingLabels = trainingSet.Labels;

    % Train multiclass SVM classifier using a fast linear solver, and set
    % 'ObservationsIn' to 'columns' to match the arrangement used for training
    % features.
    classifier = fitcecoc(trainingFeatures, trainingLabels, ...
        'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

    % Evaluate Classifier
    testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);

    % Pass CNN image features to trained classifier
    predictedLabels = predict(classifier, testFeatures);

    % Get the known labels
    testLabels = testSet.Labels;

    % Tabulate the results using a confusion matrix.
    confMat = confusionmat(testLabels, predictedLabels);

    % Convert confusion matrix into percentage form
    confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

    % Display the mean accuracy
    mean(diag(confMat))

end
