% to calculate distance among PUMAs

clear
clc
close all
shape = shaperead('michigan.shp');  % michigan
% shape = shaperead('pums_proj.shp'); % boston

shapes = {};
centroids = [];
distMatrix = [];
for i = 1 : length(shape)
    disp(['Processing shape ', num2str(i)])
    shapes{i} = polyshape(shape(i).X, shape(i).Y);
    [centroid_x, centroid_y] = centroid(shapes{i} );
    centroids(i,:) = [centroid_x, centroid_y];
end
disp('shapes ok.')
figure()
set(gcf, 'color', 'w')
for i = 1 : length(shape)
    plot( shape(i).X, shape(i).Y, 'k-')
    hold on
    plot(centroids(i,1), centroids(i,2), 'ro')
    % shape(i).PUMA = shape(i).PUMACE10;          %FOR BOSTON
    % shape(i).ALAND = shape(i).ALAND10;          %FOR BOSTON
end

k = 0;
for i = 1 : length(shape)
    for j = i+1 : length(shape)
        centroid_i = centroids(i, :);
        centroid_j = centroids(j, :);
        crtDist = sqrt(sum((centroid_i - centroid_j).^2));
        k = k + 1;
        distMatrix(k,:) = [str2num(shape(i).PUMA), str2num(shape(j).PUMA), crtDist];
        k = k + 1;
        distMatrix(k,:) = [str2num(shape(j).PUMA), str2num(shape(i).PUMA), crtDist];
    end
end
for i = 1 : length(shape)
    k = k + 1;
    crtPolyarea  = shape(i).ALAND * 1000000;
    crtInnerDist = sqrt(crtPolyarea / pi);
    distMatrix(k,:) = [str2num(shape(i).PUMA), str2num(shape(i).PUMA), crtInnerDist];
end