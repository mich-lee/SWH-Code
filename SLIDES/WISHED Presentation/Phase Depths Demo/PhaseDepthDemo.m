clear;

lambdas = [854e-9 2e-6 8e-6 20e-6];
% lambdas = [854.31e-9 854.43e-9];
% % lambda = 854.43e-9;%854.71e-9;%854.31e-9;
objectPlaneDistance = 0.5;
imagePlaneDistance = 88.23529414e-3;
M = 3000;
dx = 1.85e-6;
imageDepthRange = 10e-6; %1.4e-3; %1e-3;
targetObjectSizeSpatial = [5e-3 5e-3];

% k = 2*pi/lambda;
L = M*dx;
magnification = -imagePlaneDistance / objectPlaneDistance;


x1=-L/2:dx:L/2-dx;
[Xobj,Yobj]=meshgrid(x1,x1);

Ximg = Xobj * magnification;
Yimg = Yobj * magnification;

[Idata,cmap] = imread('northwestern.png');
Idata2_temp = ind2rgb(Idata,cmap);

targetImageSize = 2*round(targetObjectSizeSpatial / dx / 2); % [2000 2000];

Idata2_temp(:,:,1) = Idata2_temp(:,:,1) / max(max(Idata2_temp(:,:,1)));
Idata2_temp = imresize(Idata2_temp, targetImageSize);
% Idata2_temp(:,:,1) = (1 - max(min(Idata2_temp(:,:,1), 1), 0));
Idata2_temp(:,:,1) = max(min(Idata2_temp(:,:,1), 1), 0);

Idata2 = zeros([size(Xobj) 3]);
pad_x1 = ceil((size(Idata2, 2) - size(Idata2_temp, 2)) / 2);
pad_y1 = ceil((size(Idata2, 1) - size(Idata2_temp, 1)) / 2);
Idata2((pad_y1+1):(pad_y1+size(Idata2_temp,1)), (pad_x1+1):(pad_x1+size(Idata2_temp,1)), :) = Idata2_temp;

depths = Idata2(:,:,1)*imageDepthRange;
% depths(Idata2(:,:,2) == 0) = max(max(depths));
depths(Idata2(:,:,2) == 0) = 0;

% depths(:,:) = 0;
% depths(Idata2(:,:,2) == 0) = imageDepthRange;

reflectances = Idata2(:,:,2);

dists = sqrt(((Ximg - Xobj).^2) + ((Yimg - Yobj).^2) + ((depths + objectPlaneDistance + imagePlaneDistance).^2));

u = zeros(M,M,length(lambdas));
for ind = 1:length(lambdas)
	u0_temp = exp(j*2*pi/lambdas(ind)*dists) .* Idata2(:,:,2);
	u0_temp(Idata2(:,:,2) == 0) = 1e-16 * exp(-j*(pi - 1e-16)); % Force non-reflective parts to have a phase of -pi
	u(:,:,ind) = u0_temp;
end


%% Testing


fig = figure(1);
clf;
colormap('jet');

subplot(2,3,1);
plotPhasePlot(-Ximg(1,:), -Yimg(:,1), u(:,:,1), '854nm');

subplot(2,3,2);
plotPhasePlot(-Ximg(1,:), -Yimg(:,1), u(:,:,2), '2{\mu}m');

subplot(2,3,4);
plotPhasePlot(-Ximg(1,:), -Yimg(:,1), u(:,:,3), '8{\mu}m');

subplot(2,3,5);
plotPhasePlot(-Ximg(1,:), -Yimg(:,1), u(:,:,4), '20{\mu}m');

subplot(2,3,3);
imagesc(-Ximg(1,:)*1e3, -Yimg(:,1)*1e3, depths*1e6);
title('Depth', 'FontSize', 24);
xlabel("Position (mm)", 'FontSize', 20);
ylabel("Position (mm)", 'FontSize', 20);
cbar = colorbar;
cbar.Label.String = "Depth ({\mu}m)";
cbar.Label.FontSize = 20;
axis tight;
axis equal;
hold off;

sgtitle('Phases at Sensor Plane / Object Depth', 'FontSize', 30, 'FontWeight', 'bold');


% lambdaSweep = (0.2:0.01:20) * 1e-6;
lambdaSweep = round(logspace(-7, -5, 32), 8);
exportImages = cell(1, length(lambdaSweep));
hold on;
for ind = 1:length(lambdaSweep)
	u0_temp = exp(j*2*pi/lambdaSweep(ind)*dists) .* Idata2(:,:,2);
	u0_temp(Idata2(:,:,2) == 0) = 1e-16 * exp(-j*(pi - 1e-16)); % Force non-reflective parts to have a phase of -pi

	subplot(2,3,6);
	plotPhasePlot(-Ximg(1,:), -Yimg(:,1), u0_temp, [num2str(lambdaSweep(ind)*1e6) '{\mu}m']);

	drawnow;

	frame = getframe(fig);
    exportImages{ind} = frame2im(frame);

% 	pause;
end
hold off;


fps = 2;
filename = "testAnimated.gif"; % Specify the output file name
for idx = 1:length(exportImages)
    [A,map] = rgb2ind(exportImages{idx},256);
    if idx == 1
        imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",1/fps);
    else
        imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",1/fps);
    end
end


%% Functions
function plotPhasePlot(xCoords, yCoords, u, lambdaStr)
	imagesc(xCoords*1e3, yCoords*1e3, angle(u));
	title({["\lambda = " + lambdaStr]}, 'FontSize', 24);
	xlabel("Position (mm)", 'FontSize', 20);
	ylabel("Position (mm)", 'FontSize', 20);
	cbar = colorbar;
	cbar.Label.String = "Phase (radians)";
	cbar.Label.FontSize = 20;
	caxis([-pi pi]);
	axis tight;
	axis equal;
	hold off;
end