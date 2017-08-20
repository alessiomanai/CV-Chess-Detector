% train RCNN

%scacchi = load('labelScacchi.mat'); %carico una tabella con il modello

%Nvidia GPU error
%rcnn = trainFasterRCNNObjectDetector(scacchi, layers, opts); 

rcnnNet = trainRCNNObjectDetector(scacchi, layers, opts);

img = imread('riconoscimento/prova.jpg');

[bbox, score, label] = detect(rcnnNet, img, 'MiniBatchSize', 32);

[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure, imshow(detectedImg);
