clear;
addpath('matplotlib/');

freq = 440;
coords = (-7:7)/6;
fieldPhasorX = 1; %1;
fieldPhasorY = 1j; %3*exp(1j*pi/4)*fieldPhasorX;


xCoords = coords;
yCoords = coords;
[xGrid, yGrid] = meshgrid(xCoords, yCoords);
delta = xCoords(2) - xCoords(1);

fieldPhasorTempMag = sqrt((abs(fieldPhasorX).^2) + (abs(fieldPhasorY).^2));
fieldPhasorX = fieldPhasorX / fieldPhasorTempMag;
fieldPhasorY = fieldPhasorY / fieldPhasorTempMag;

field = zeros([size(xGrid) 2]);
field(:,:,1) = fieldPhasorX;
field(:,:,2) = fieldPhasorY;

depthPhaseShift = exp(1j * sqrt(xGrid.^2 + yGrid.^2) * 2*pi/2);
field = field .* depthPhaseShift;

field(:,1,:) = 0;
field(:,end,:) = 0;
field(1,:,:) = 0;
field(end,:,:) = 0;

field = 0.65 * delta * field / max(max(max(abs(field))));

T = 1 / freq;
tEnd = 1 * T;
t = 0:(T/48):tEnd;
t = t(1:end-1);



figure(1);
clf;

% subplot(1, 2, 1);
% u = real(field(2:end-1,2:end-1,1));
% v = real(field(2:end-1,2:end-1,2));
% quiverHandle0 = quiver(xGrid(2:end-1,2:end-1), yGrid(2:end-1,2:end-1), u, v, 'off', 'Marker', '.', 'ShowArrowHead', 'on', 'LineWidth', 1, 'Color', 'white', 'SeriesIndex', 1);
% set(quiverHandle0, 'Color', 'white');
% formatPlot(xGrid, yGrid, '');

% subplot(1, 2, 2);
exportImages = cell(1,length(t));
for ind = 1:length(t)
	phaseShift = exp(-1j*2*pi*freq*t(ind));
	u = real(field(2:end-1,2:end-1,1) .* phaseShift);
	v = real(field(2:end-1,2:end-1,2) .* phaseShift);

	if (ind ~= 1)
		delete(quiverHandle);
	end

	quiverHandle = quiver(xGrid(2:end-1,2:end-1), yGrid(2:end-1,2:end-1), u, v, 'off', 'Marker', '.', 'ShowArrowHead', 'on', 'LineWidth', 1, 'Color', 'white', 'SeriesIndex', 1);
	formatPlot(xGrid, yGrid, '');

	drawnow;

	exportImages{1, ind} = captureFigureImage(1);

% 	pause(0.5);
end


fps = floor(length(t) / 2);
filename = "polarized_field.gif";
for idx = 1:size(exportImages, 2)
	[A,map] = rgb2ind(exportImages{1,idx},256);
	if idx == 1
    	imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",1/fps);
	else
    	imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",1/fps);
	end
end




%% Functions
function formatPlot(xGrid, yGrid, plotTitle)
	set(gca,'Color','black');
	set(gca,'XTick',[])
	set(gca,'YTick',[])
	axis equal;
	axis tight;
	xlim([xGrid(1,1) xGrid(end,end)]);
	ylim([yGrid(1,1) yGrid(end,end)]);
	title(plotTitle, 'FontSize', 30);
end

function img = captureFigureImage(figNum)
	fig = figure(figNum);
	drawnow;
	frame = getframe(fig);
	img = frame2im(frame);
end