
function riconosci(immagine)

    I = imread('ima4.jpg');
    starting = rgb2gray(I);
    
    starting = imadjust(starting);
    
    figure, imshow(starting);
    puntiStarting = detectSURFFeatures(starting);

    regina = imread(immagine);
    regina = rgb2gray(regina);
    
    figure, imshow(regina);
    regina = imadjust(regina); %soluzione migliore
    %regina = histeq(regina);
    %figure, imshow(regina);
    
    puntiRegina = detectSURFFeatures(regina);
    
    figure;
    imshow(regina);
    title('100 Strongest Feature Points from Box Image');
    hold on;
    plot(selectStrongest(puntiRegina, 100));

    [reginaFeat, puntiRegina] = extractFeatures(regina, puntiRegina);
    [startingFeat, puntiStarting] = extractFeatures(starting, puntiStarting);
    
    reginaCoppie = matchFeatures(reginaFeat, startingFeat);
    
    matchedReginaPoints = puntiRegina(reginaCoppie(:, 1), :);
    matchedStartingPoints = puntiStarting(reginaCoppie(:, 2), :);
    figure;
    showMatchedFeatures(regina, starting, matchedReginaPoints, ...
        matchedStartingPoints, 'montage');
    title('Putatively Matched Points (Including Outliers)');
    
    [tform, inlierBoxPoints, inlierStartingPoints] = ...
    estimateGeometricTransform(matchedReginaPoints, matchedStartingPoints, 'affine');

    figure;
    showMatchedFeatures(regina, starting, inlierBoxPoints, ...
        inlierStartingPoints, 'montage');
    title('Matched Points (Inliers Only)');
    
    scacchiPolygon = [1, 1;...                           % top-left
        size(regina, 2), 1;...                 % top-right
        size(regina, 2), size(regina, 1);... % bottom-right
        1, size(regina, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon
    
    newScacchiPolygon = transformPointsForward(tform, scacchiPolygon);
    
    figure;
    imshow(starting);
    hold on;
    line(newScacchiPolygon(:, 1), newScacchiPolygon(:, 2), 'Color', 'y');
    title('Detected Box');

