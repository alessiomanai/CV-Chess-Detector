
[net, featureLayer, classifier] = DeepLearningImageClassification

for i=1:14
    numero = int2str(i);
    nome = strcat('detectionnew/', numero,'.jpg')
    I = imread(nome);
    img = preprocessImage(I);
    imageFeatures = activations(net, img, featureLayer);
    label = predict(classifier, imageFeatures)
end