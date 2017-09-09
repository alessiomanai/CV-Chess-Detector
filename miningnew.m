setDir  = fullfile('pezzinew');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

[trainingSet, testSet] = splitEachLabel(imds, 0.4, 'randomize');    %percentuale corrisponde alle immagini prese come training

bag = bagOfFeatures(trainingSet, 'GridStep', [8 8], 'VocabularySize', 1000);

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

confMatrix = evaluate(categoryClassifier, trainingSet);

confMatrix = evaluate(categoryClassifier, testSet);

fprintf('regina nera');
img = imread('detectionnew/1.jpg'); %regina nera
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('cavallo bianco');
img = imread('detectionnew/2.jpg'); %cavallo bianco 
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('pedone bianco');
img = imread('detectionnew/3.jpg'); %pedone bianco
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('cavallo nero');
img = imread('detectionnew/4.jpg'); %cavallo nero
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('regina bianca');
img = imread('detectionnew/5.jpg'); %regina bianca
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('cavallo bianco');
img = imread('detectionnew/6.jpg'); %cavallo bianco
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('regina nera');
img = imread('detectionnew/7.jpg'); %regina nera
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('regina bianca')
img = imread('detectionnew/8.jpg'); %regina bianca
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('cavallo nero');
img = imread('detectionnew/9.jpg'); %cavallo nero
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

fprintf('pedone bianco');
img = imread('detectionnew/14.jpg'); %pedone bianco
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)

