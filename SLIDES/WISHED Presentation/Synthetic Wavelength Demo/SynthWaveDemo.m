clear;
addpath('matplotlib/');

lambda = [1e-3 1.2e-3]; %[854.31e-9 854.43e-9];

lambdaSynth = lambda(1) * lambda(2) / abs(lambda(2) - lambda(1));

xmin = -10e-3;
xmax = 10e-3;
ymin = 0;
ymax = 20e-3;
deltaX = 25e-6;
deltaY = 25e-6;

xCoords = linspace(xmin, xmax, ceil((xmax - xmin) / deltaX));
yCoords = linspace(ymin, ymax, ceil((ymax - ymin) / deltaY));

plotCutoffY = 5e-3;

apertureNormal = [0 1 0];

numElems = 500;
elementLocs = [transpose(1*(-ceil((numElems-1)/2):floor((numElems-1)/2))/numElems*(xmax-xmin)) transpose(repelem(ymin, numElems)) zeros(numElems,1)];
% elementLocs(:,2) = ymin + (1e-3)*0.5*(1 + cos(2*pi*elementLocs(:,1) / (max(xCoords) - min(xCoords))));


k = zeros(1, 1, length(lambda));
k(1,1,:) = 2*pi ./ lambda;

% inputVector = ones(numElems,1,length(lambda));
inputVector = exp(1j*(sqrt(sum(([0 17.5e-3 0] - elementLocs).^2, 2)) .* k));
inputVector = inputVector ./ sqrt(sum(abs(inputVector).^2, 1)) / numElems;

fieldsTemp = get_field_patterns(lambda, 1, elementLocs, inputVector, xCoords, yCoords, 0, false, apertureNormal);

fields = fieldsTemp;

synthField = fields(:,:,1) .* conj(fields(:,:,2));
synthField = sqrt(abs(synthField)) .* exp(j*angle(synthField));



% fieldToPlot = fields(:,:,1);
% fieldToPlot = synthField;
% plotMaxAbs = max(max(abs(fieldToPlot(yCoords>plotCutoffY,:,:))));
% 
% figure(1);
% clf;
% colormap('twilight');
% subplot(1,2,1);
% imagesc(xCoords*1000, yCoords*1000, abs(fieldToPlot));
% colorbar;
% caxis([0 plotMaxAbs]);
% set(gca, 'YDir', 'normal');
% axis equal;
% axis tight;
% subplot(1,2,2);
% imagesc(xCoords*1000, yCoords*1000, angle(fieldToPlot));
% colorbar;
% caxis([-pi pi]);
% set(gca, 'YDir', 'normal');
% axis equal;
% axis tight;





v_prop = 1 / sqrt((8.854e-12)*(4*pi*(1e-7)));
freqs = v_prop ./ (2*pi ./ k);
T = 1 ./ freqs;
T_synth_wavelen = 1 / (v_prop / lambdaSynth);
tStep = min(min(min(T))) / 12;
tEnd = 1*T_synth_wavelen;
t = linspace(0, tEnd, ceil(tEnd / tStep));
t = t(1:end-1);
exportImages = cell(4, length(t));
figure(1);
clf;
stylePhasePlot('\lambda = 1.00mm');
hold on;
figure(2);
clf;
stylePhasePlot('\lambda = 1.20mm');
hold on;
figure(3);
clf;
stylePhasePlot('\Lambda = 6.00mm');
hold on;
figure(4);
stylePhasePlot('Superimposed (\lambda = 1.00mm, 1.20mm)', [1 0 0; 0 1 0]);
% stylePhasePlot('', [transpose(linspace(1, 0, 100)) transpose(linspace(0, 1, 100)) zeros(100, 1)]);
hold on;
for ind = 1:length(t)
	u_temp = fields .* exp(1j*2*pi*freqs*t(ind));
	u_synth_temp = u_temp(:,:,1) .* conj(u_temp(:,:,2));

	alphaLim = 1;
	alphaThreshold = 0.3;
	alphaCutoffSteepness = 20;
	alphaMask1 = getAlphaMask(u_temp(:,:,1), alphaLim, alphaThreshold, alphaCutoffSteepness);
	alphaMask2 = getAlphaMask(u_temp(:,:,2), alphaLim, alphaThreshold, alphaCutoffSteepness);
	alphaMask3 = getAlphaMask(sqrt(abs(u_temp(:,:,1)) .* abs(u_temp(:,:,2))), alphaLim, alphaThreshold, alphaCutoffSteepness);

	figure(1);
	cla;
	imagesc(xCoords*1000, yCoords*1000, angle(u_temp(:,:,1)), 'AlphaData', alphaMask1);
	figure(2);
	cla;
	imagesc(xCoords*1000, yCoords*1000, angle(u_temp(:,:,2)), 'AlphaData', alphaMask2);
	figure(3);
	cla;
	imagesc(xCoords*1000, yCoords*1000, angle(u_synth_temp), 'AlphaData', alphaMask3);
	
	figure(4);
	cla;
	hold on;
% 	imagesc(xCoords*1000, yCoords*1000, angle(u_temp(:,:,1)), 'AlphaData', alphaMask1 * 0.5);
% 	imagesc(xCoords*1000, yCoords*1000, angle(u_temp(:,:,2)), 'AlphaData', alphaMask2 * 0.5);
	tempAngle1 = pi/2*2*((angle(u_temp(:,:,1)) >= 0) - 0.5);
	tempAngle2 = pi/2*2*((angle(u_temp(:,:,2)) >= 0) - 0.5);
	imagesc(xCoords*1000, yCoords*1000, tempAngle1, 'AlphaData', alphaMask1 * 0.5);
	imagesc(xCoords*1000, yCoords*1000, tempAngle2, 'AlphaData', alphaMask2 * 0.5);
	hold off;

	
	drawnow;

	for m = 1:4
		fig = figure(m);
		frame = getframe(fig);
		exportImages{m,ind} = frame2im(frame);
	end

% 	pause;
% 	pause(0.1);
end
% hold off;

%% Exporting images
fps = floor(length(t) / 10);
for m = 1:size(exportImages, 1)
	for idx = 1:size(exportImages, 2)
		filename = "output" + num2str(m) + ".gif";
    	[A,map] = rgb2ind(exportImages{m,idx},256);
    	if idx == 1
        	imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",1/fps);
    	else
        	imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",1/fps);
    	end
	end
end





%% Functions
function stylePhasePlot(titleStr, varargin)
	cbar = colorbar;
	cbar.Label.String = 'Phase (radians)';
	cbar.Label.FontSize = 16;
	caxis([-pi pi]); set(gca, 'YDir', 'normal');
	axis equal; axis tight;
	xlabel('Position (mm)', 'FontSize', 16);
	ylabel('Position (mm)', 'FontSize', 16);
	title(titleStr, 'FontSize', 20);
	if isempty(varargin)
		colormap('inferno');
	else
		colormap(varargin{1});
	end
	set(gca, 'Color', 'black');
end

function alpha = getAlphaMask(u, upperLim, lowerCutoff, cutoffSteepness)
	alpha = min(abs(u(:,:,1)), upperLim);

	% This is a sigmoid
	cutoffMask = 1 ./ (1 + exp(-cutoffSteepness*(alpha - lowerCutoff)));

	alpha = alpha .* cutoffMask;
end

function [g, xGrid, yGrid] = get_field_patterns(lambda, n_medium, elementLocs, inputVector, xCoords, yCoords, zCut, useObliquity, apertureNormal)
	errFlagFocusingVector = false;
	if (length(size(inputVector)) == 2)
		if (~iscolumn(inputVector))
			errFlagFocusingVector = true;
		end
	elseif (length(size(inputVector)) == 3)
		if (size(inputVector, 2) ~= 1)
			errFlagFocusingVector = true;
		end
	else
		errFlagFocusingVector = true;
	end
	if (errFlagFocusingVector)
		error("'inputVector' should be a column vector or an Nx1xM vector with the last dimension corresponding to wavelength.");
	end

	if (~isvector(lambda))
		error("'lambda0' should be a vector or a scalar.")
	end
	if (~isvector(n_medium))
		error("'n_medium' should be a vector or a scalar.")
	end

	[xGrid, yGrid] = meshgrid(xCoords, yCoords);
	g = zeros([size(xGrid) length(lambda)]);

	k_lambda = n_medium .* (2*pi ./ lambda);
	k_lambda = gpuArray(reshape(k_lambda, [1 1 1 length(k_lambda)]));
	
	dx = gpuArray(xGrid - reshape(elementLocs(:,1), [1 1 size(elementLocs, 1)]));
	dy = gpuArray(yGrid - reshape(elementLocs(:,2), [1 1 size(elementLocs, 1)]));
	dz = gpuArray(zCut - reshape(elementLocs(:,3), [1 1 size(elementLocs, 1)]));
	R = sqrt((dx.^2) + (dy.^2) + (dz.^2));

	if (useObliquity)
		apertureNorm = apertureNormal / sqrt(sum(apertureNormal .^ 2));
		obliquity = ((dx * apertureNorm(1)) + (dy * apertureNorm(2)) + (dz * apertureNorm(3))) ./ R;
	else
		obliquity = 1;
	end

	clear("dx", "dy", "dz");
	wait(gpuDevice);

	for m = 1:length(k_lambda)
		focusingVecTemp = gpuArray(reshape(inputVector(:,1,m), [1 1 size(inputVector, 1) 1]));
		g(:,:,m) = sum(focusingVecTemp .* (exp(-j*k_lambda(1,1,1,m).*R) ./ R) .* obliquity, 3);
	end
end