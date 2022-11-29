clear;

lambdas = [854.31e-9 854.43e-9];
% lambda = 854.43e-9;%854.71e-9;%854.31e-9;
objectPlaneDistance = 0.5;
imagePlaneDistance = 88.23529414e-3;
M = 8000;
dx = 5e-6;
% imageDepthRange = 1.4e-3; %1e-3;
targetObjectSizeSpatial = [30e-3 30e-3];

% k = 2*pi/lambda;
L = M*dx;


x1=-L/2:dx:L/2-dx;
[X1,Y1]=meshgrid(x1,x1);

if 0
	[Idata,cmap] = imread('../../SWH Code/northwestern.png');
	Idata2 = ind2rgb(Idata,cmap);

	imageDepthRange = 1.4e-3; %1e-3;
	targetImageSize = 2*round(targetObjectSizeSpatial / dx / 2); % [2000 2000];
elseif 0
	[Idata,cmap] = imread('../../SWH Code/teapot.png');
	Idata2_temp = ind2rgb(Idata,cmap);
	Idata2_temp = Idata2_temp(:,201:1200,:);
	Idata2 = zeros(1300, 1300, 3);
	Idata2(:,:,1) = 0.192156862745098;
	Idata2(151:1150, 151:1150, :) = Idata2_temp;
	Idata2 = Idata2 - 0.192156862745098;

	imageDepthRange = L * (27.2/40) * (4/6.4);
	targetImageSize = [M M];
elseif 1
	[Idata,cmap] = imread('../../SWH Code/bunny.png');
	
% 	[gridTempX, gridTempY] = meshgrid(-16:16, -16:16);
% 	filtTemp = exp(-(1/2)*(gridTempX.^2 + gridTempY.^2)/(10^2));
	n = 256;
	m = 128;
	filtTemp = zeros(n, n);
	filtTemp((((n-m)/2)+1):(((n-m)/2)+m),1:end) = repmat([1:(n/2) (n/2):-1:1] / (n/2), m, 1);
	filtTemp = min(filtTemp + filtTemp.', 1);

	imageDepthRange = L * (27.2/40) * (4/6.4);
	targetImageSize = [M M];

	Idata2_temp = ind2rgb(Idata,cmap);
	Idata2 = zeros(2500, 2500, 3);
	Idata2(251:2250, 251:2250, :) = Idata2_temp;
	Idata2_temp2 = filter2(filtTemp, Idata2(:,:,1) .* (Idata2(:,:,2) == 1));
	Idata2(:,:,1) = (imageDepthRange / L * 2000) * Idata2_temp2 / max(max(Idata2_temp2));
end
Idata2(:,:,1) = Idata2(:,:,1) / max(max(Idata2(:,:,1)));
Idata2 = imresize(Idata2, targetImageSize);
Idata2 = min(Idata2, 1);

depths = Idata2(:,:,1)*imageDepthRange;

if 1
	lambdas = imageDepthRange;
end

u0 = zeros(M,M,length(lambdas));
for ind = 1:length(lambdas)
	u0_temp = exp(j*2*pi/lambdas(ind)*depths) .* Idata2(:,:,2);
	startInds = floor((size(u0, [1 2]) - size(u0_temp)) / 2) + 1;
	endInds = startInds + size(u0_temp) - 1;
	u0(startInds(1):endInds(1), startInds(2):endInds(2), ind) = u0_temp;
end


[~,~,nz] = surfnorm(X1,Y1,depths);
fakeFieldCollection = u0 .* nz;


%% Testing
fakeFieldSelectInd = 1;

subsamplingFactor = 350;
startInd = floor(mod(size(u0, 2), subsamplingFactor) / 2) + 1;
X2 = X1(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end);
Y2 = Y1(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end);
fakeField = fakeFieldCollection(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end,fakeFieldSelectInd);
fakeField = fakeField + 0.25*rand(size(fakeField)).*exp(j*2*pi*rand(size(fakeField)));

subsamplingFactor2 = 50;
startInd = floor(mod(size(u0, 2), subsamplingFactor2) / 2) + 1;
X3 = X1(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end);
Y3 = Y1(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end);
plotOverlayField = fakeFieldCollection(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end,fakeFieldSelectInd);
	% plotOverlayField = abs(plotOverlayField) .^ 2;
gridTempSize = 128;
[gridTempX, gridTempY] = meshgrid(-(gridTempSize/2):(gridTempSize/2), -(gridTempSize/2):(gridTempSize/2));
filtTemp = exp(-(1/2)*(gridTempX.^2 + gridTempY.^2)/(3^2));
plotOverlayField = filter2(filtTemp, plotOverlayField);
	% imagesc(X3(1,:), Y3(:,1), abs(plotOverlayField));
	% axis equal;
	% axis tight;

subsamplingFactor3 = 50;
startInd = floor(mod(size(u0, 2), subsamplingFactor3) / 2) + 1;
X4 = X1(startInd:subsamplingFactor3:end,startInd:subsamplingFactor3:end);
Y4 = Y1(startInd:subsamplingFactor3:end,startInd:subsamplingFactor3:end);
phaseOverlay = fakeFieldCollection(startInd:subsamplingFactor3:end,startInd:subsamplingFactor3:end,fakeFieldSelectInd);

% clf;
% quiver(X2,Y2,zeros(size(X2)),abs(fakeField), 0.6, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 1);
% set(gca,'Color','black');
% axis equal;
% axis tight;

phi = 2*pi*(0:5/360:1);

if (phi(end) == 2*pi)
	phi = phi(1:end-1);
end

fig = figure(12);
clf;
colormap('bone');

plotOverlayFieldIm = imagesc(X3(1,:), Y3(:,1), abs(plotOverlayField));
plotOverlayFieldIm.AlphaData = 0.5;

exportImages = {};
hold on;
for ind = 1:length(phi)
	curPhaseShift = exp(-j*phi(ind));
	tempFakeField = fakeField * curPhaseShift; % Using the convention that phasors spin clockwise in time
	mag = 2*real(tempFakeField);
	
	if (ind ~= 1)
% 		delete(phaseOverlayIm);
		delete(quiverHandle);
	end

% 	phaseOverlayIm = imagesc(X4(1,:), Y4(:,1), angle((phaseOverlay*curPhaseShift) + 0.25*rand(size(phaseOverlay)).*exp(j*2*pi*rand(size(phaseOverlay)))), 'Interpolation', 'bilinear');
% 	phaseOverlayIm.CDataMapping = "direct";
% 	phaseOverlayIm.CData = (((phaseOverlayIm.CData / max(max(abs(phaseOverlayIm.CData)))) / 2) + 0.5) * 255;
	
	theta = 80*(pi/180);
	u = mag*cos(theta);
	v = -mag*sin(theta);

	quiverHandle = quiver(X2, Y2, u, v, 1.5, 'Marker', '.', 'ShowArrowHead', 'on', 'LineWidth', 0.5, 'Color', 'white', 'SeriesIndex', 1);
	set(gca,'Color','black');
	set(gca,'XTick',[])
	set(gca,'YTick',[])
	axis equal;
	axis tight;
	xlim([X3(1,1) X3(end,end)]);
	ylim([Y3(1,1) Y3(end,end)]);

	drawnow;

	frame = getframe(fig);
    exportImages{ind} = frame2im(frame);

% 	pause;
end
hold off;


fps = 24;
filename = "testAnimated.gif"; % Specify the output file name
for idx = 1:length(exportImages)
    [A,map] = rgb2ind(exportImages{idx},256);
    if idx == 1
        imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",1/fps);
    else
        imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",1/fps);
    end
end


% figure(1);
% clf;
% imagesc([x1(1) x1(end)]*1000, [x1(1) x1(end)]*1000, Idata2(:,:,1));
% axis equal;
% axis tight;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);
% colorbar;
% 
% figure(2);
% clf;
% imagesc([x1(1) x1(end)]*1000, [x1(1) x1(end)]*1000, depths*1000);
% axis equal;
% axis tight;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);
% colorbar;

% figure(3);
% clf;
% subsamplingFactor = 32;
% surf(X1(1:subsamplingFactor:end, 1:subsamplingFactor:end), -Y1(1:subsamplingFactor:end, 1:subsamplingFactor:end), depths(1:subsamplingFactor:end, 1:subsamplingFactor:end), ...
% 		EdgeAlpha=0);
% axis equal;
% axis tight;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);
% colorbar;
% view(2);