
[net, featureLayer, classifier] = DeepLearningImageClassification

%codice

I = imread('riconoscimento3.jpg');
I = rgb2gray(I);
I = imadjust(I);

[M N] = size(I);    
A = I(1:N/2, 1:N/2);
B = I(1:M/2, N/2+1:N);
C = I(M/2+1:M, 1:N/2);
D = I(M/2+1:M, N/2+1:N);

[A1 B1 C1 D1] = dividiScena(A);
[A2 B2 C2 D2] = dividiScena(B);
[A3 B3 C3 D3] = dividiScena(C);
[A4 B4 C4 D4] = dividiScena(D);

img = preprocessImage(A1);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A2);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A3);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A4);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(B1);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B2);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B3);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B4);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(C1);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C2);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C3);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C4);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(D1);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D2);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D3);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D4);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)