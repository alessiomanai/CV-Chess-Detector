%follia 64 quadranti

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

% prendo tutti i quadranti (follia)
[A11 B11 C11 D11] = dividiScena(A1);
[A12 B12 C12 D12] = dividiScena(B1);
[A13 B13 C13 D13] = dividiScena(C1);
[A14 B14 C14 D14] = dividiScena(D1);

[A21 B21 C21 D21] = dividiScena(A2);
[A22 B22 C22 D22] = dividiScena(B2);
[A23 B23 C23 D23] = dividiScena(C2);
[A24 B24 C24 D24] = dividiScena(D2);

[A31 B31 C31 D31] = dividiScena(A3);
[A32 B32 C32 D32] = dividiScena(B3);
[A33 B33 C33 D33] = dividiScena(C3);
[A34 B34 C34 D34] = dividiScena(D3);

[A41 B41 C41 D41] = dividiScena(A4);
[A42 B42 C42 D42] = dividiScena(B4);
[A43 B43 C43 D43] = dividiScena(C4);
[A44 B44 C44 D44] = dividiScena(D4);

%[net, featureLayer, classifier] = DeepLearningImageClassification


img = preprocessImage(A11);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A12);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A13);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A14);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(B11);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B12);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B13);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B14);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(C11);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C12);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C13);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C14);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(D11);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D12);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D13);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D14);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)




img = preprocessImage(A21);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A22);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A23);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A24);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(B21);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B22);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B23);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B24);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(C21);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C22);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C23);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C24);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(D21);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D22);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D23);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D24);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)



img = preprocessImage(A31);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A32);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A33);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A34);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(B31);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B32);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B33);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B34);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(C31);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C32);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C33);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C34);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(D31);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D32);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D33);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D34);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)



img = preprocessImage(A41);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A42);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A43);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(A44);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(B41);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B42);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B43);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(B44);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(C41);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C42);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C43);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(C44);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)


img = preprocessImage(D41);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D42);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D43);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)

img = preprocessImage(D44);
imageFeatures = activations(net, img, featureLayer);
label = predict(classifier, imageFeatures)