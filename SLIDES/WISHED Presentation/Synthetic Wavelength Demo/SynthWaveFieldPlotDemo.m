clear;
addpath('matplotlib/');

lambdas = [1e-3 1.2e-3];
% lambda = 854.43e-9;%854.71e-9;%854.31e-9;
objectPlaneDistance = 0.5;
imagePlaneDistance = 88.23529414e-3;
M = 1000;
dx = 10e-6;
% imageDepthRange = 1.4e-3; %1e-3;
targetObjectSizeSpatial = [30e-3 30e-3];

% k = 2*pi/lambda;
L = M*dx;


x1=-L/2:dx:L/2-dx;
[X1,Y1]=meshgrid(x1,x1);

lambdaSynth = lambdas(1) * lambdas(2) / abs(lambdas(2) - lambdas(1));

k = zeros(1, 1, length(lambdas));
k(1,1,:) = 2*pi ./ lambdas;


[Idata,cmap] = imread('bunny.png');

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

Idata2(:,:,1) = Idata2(:,:,1) / max(max(Idata2(:,:,1)));
Idata2 = imresize(Idata2, targetImageSize);
Idata2 = min(Idata2, 1);

depths = Idata2(:,:,1)*imageDepthRange;

% lambdas = imageDepthRange;

u0 = zeros(M,M,length(lambdas));
for ind = 1:length(lambdas)
	u0_temp = exp(j*2*pi/lambdas(ind)*depths) .* Idata2(:,:,2);
	startInds = floor((size(u0, [1 2]) - size(u0_temp)) / 2) + 1;
	endInds = startInds + size(u0_temp) - 1;
	u0(startInds(1):endInds(1), startInds(2):endInds(2), ind) = u0_temp;
end

[~,~,nz] = surfnorm(X1,Y1,depths);
u0 = u0 .* nz;

% uSynth = u0(:,:,1) .* conj(u0(:,:,2));
% u0(:,:,3) = uSynth;



subsamplingFactor = 50;
startInd = floor(mod(size(u0, 2), subsamplingFactor) / 2) + 1;
X2 = X1(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end);
Y2 = Y1(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end);
plotFields = u0(startInd:subsamplingFactor:end,startInd:subsamplingFactor:end,:);
plotFields = plotFields + 0.1*rand(size(plotFields)).*exp(j*2*pi*rand(size(plotFields))).*(abs(plotFields) < 0.1);

subsamplingFactor2 = 1;
filterGridTempSize = 128;
startInd = floor(mod(size(u0, 2), subsamplingFactor2) / 2) + 1;
X3 = X1(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end);
Y3 = Y1(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end);
plotOverlayFields = u0(startInd:subsamplingFactor2:end,startInd:subsamplingFactor2:end,:);
[gridTempX, gridTempY] = meshgrid(-(filterGridTempSize/2):(filterGridTempSize/2), -(filterGridTempSize/2):(filterGridTempSize/2));
filtTemp = exp(-(1/2)*(gridTempX.^2 + gridTempY.^2)/(30^2));
for idx = 1:size(plotOverlayFields, 3)
	plotOverlayFields(:,:,idx) = filter2(filtTemp, abs(plotOverlayFields(:,:,idx)));
end


%% Plotting
v_prop = 1 / sqrt((8.854e-12)*(4*pi*(1e-7)));
freqs = v_prop ./ (2*pi ./ k);
T = 1 ./ freqs;
T_synth_wavelen = 1 / (v_prop / lambdaSynth);
tStep = min(min(min(T))) / 12;
tEnd = 1*T_synth_wavelen;
t = linspace(0, tEnd, ceil(tEnd / tStep));
t = t(1:end-1);

for idx = 1:4
	figure(idx);
	clf;
	colormap('bone');

	% Arbitrarily select the first wavelength's field
	plotOverlayFieldIm = imagesc(X3(1,:), Y3(:,1), abs(plotOverlayFields(:,:,1)));
	plotOverlayFieldIm.AlphaData = 0.6;
end

exportImages = cell(4,length(t));
for ind = 1:length(t)
	curPhaseShift = exp(-j*2*pi*freqs*t(ind));
	tempFakeField = plotFields .* curPhaseShift; % Using the convention that phasors spin clockwise in time
	mag = 2*real(tempFakeField);

	magSynth = 2*real(tempFakeField(:,:,1) .* conj(tempFakeField(:,:,2)));
	
	if (ind ~= 1)
		delete(quiverHandle1);
		delete(quiverHandle2);
		delete(quiverHandle1b);
		delete(quiverHandle2b);
		delete(quiverHandle3);
	end
	
	theta = 80*(pi/180);
	uComps = mag*cos(theta);
	vComps = -mag*sin(theta);
	uCompSynth = magSynth*cos(theta);
	vCompSynth = -magSynth*sin(theta);

	u1 = uComps(:,:,1);
	v1 = vComps(:,:,1);
	u2 = uComps(:,:,2);
	v2 = vComps(:,:,2);

	figure(1);
	hold on;
	quiverHandle1 = quiver(X2, Y2, u1, v1, 0.75, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 2, 'Color', 'white', 'SeriesIndex', 1);
	set(quiverHandle1, 'Color', 'red');
	formatPlot(X3, Y3, '\lambda = 1.00mm');
	hold off;
	exportImages{1, ind} = captureFigureImage(1);

	figure(2);
	hold on;
	quiverHandle2 = quiver(X2, Y2, u2, v2, 0.75, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 1, 'Color', 'white', 'SeriesIndex', 1);
	set(quiverHandle2, 'Color', 'green');
	formatPlot(X3, Y3, '\lambda = 1.20mm');
	hold off;
	exportImages{2, ind} = captureFigureImage(2);

	figure(3);
	hold on;
	quiverHandle1b = quiver(X2, Y2, u1, v1, 0.75, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 2, 'Color', 'white', 'SeriesIndex', 1);
	set(quiverHandle1b, 'Color', 'red');
	quiverHandle2b = quiver(X2, Y2, u2, v2, 0.75, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 1, 'Color', 'white', 'SeriesIndex', 1);
	set(quiverHandle2b, 'Color', 'green');
	formatPlot(X3, Y3, '\lambda = 1.00mm, 1.20mm');
	hold off;
	exportImages{3, ind} = captureFigureImage(3);

	figure(4);
	hold on;
	quiverHandle3 = quiver(X2, Y2, uCompSynth, vCompSynth, 0.75, 'Marker', '.', 'ShowArrowHead', 'off', 'LineWidth', 2, 'Color', 'white', 'SeriesIndex', 1);
	set(quiverHandle3, 'Color', 'white');
	formatPlot(X3, Y3, '\Lambda = 6.00mm');
	hold off;
	exportImages{4, ind} = captureFigureImage(4);

	drawnow;

% 	pause;
end

fps = floor(length(t) / 10);
for m = 1:size(exportImages, 1)
	filename = "field" + num2str(m) + ".gif";
	for idx = 1:size(exportImages, 2)
    	[A,map] = rgb2ind(exportImages{m,idx},256);
		if idx == 1
        	imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",1/fps);
    	else
        	imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",1/fps);
		end
	end
end



%% Functions
function formatPlot(X3, Y3, plotTitle)
	set(gca,'Color','black');
	set(gca,'XTick',[])
	set(gca,'YTick',[])
	axis equal;
	axis tight;
	xlim([X3(1,1) X3(end,end)]);
	ylim([Y3(1,1) Y3(end,end)]);
	title(plotTitle, 'FontSize', 30);
end

function img = captureFigureImage(figNum)
	fig = figure(figNum);
	drawnow;
	frame = getframe(fig);
	img = frame2im(frame);
end