% train RCNN

%scacchi = load('labelScacchi.mat'); %carico una tabella con il modello

%Nvidia GPU error
%rcnn = trainFasterRCNNObjectDetector(scacchi, layers, opts); 

rcnnNet = trainRCNNObjectDetector(scacchi, layers, opts);

img = imread('riconoscimento/prova.jpg');

[bbox, score, label] = detect(rcnnNet, img, 'MiniBatchSize', 32);

n = numel(label);

annotation = sprintf('%s: (Confidence = %f)', label(1), score(1));
detectedImg = insertObjectAnnotation(img, 'rectangle', bbox(1,:), annotation);
    
for i=2:n
    annotation = sprintf('%s: (Confidence = %f)', label(i), score(i));
    detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(i,:), annotation);
end

figure, imshow(detectedImg);
