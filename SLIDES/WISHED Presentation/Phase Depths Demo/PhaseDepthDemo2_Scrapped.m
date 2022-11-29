clear;

lambda = [854.31e-9 854.35e-9]; %[854.31e-9 854.43e-9];

xmin = -0.2;
xmax = 0.5*3/17;
ymin = -0.001;%-0.052/2;
ymax = 0.001;%0.052/2;
deltaX = 1e-3;
deltaY = 3.2e-6;

focalLen = 75e-3;
lensLocX = 0;

xCoords = linspace(xmin, xmax, ceil((xmax - xmin) / deltaX));
yCoords = linspace(ymin, ymax, ceil((ymax - ymin) / deltaY));
apertureNormal = [1 0 0];

numElems = 10;
elementLocs = [transpose(repelem(-0.5, numElems)) transpose((-ceil((numElems-1)/2):floor((numElems-1)/2))/numElems*(ymax-ymin)) zeros(numElems,1)];
% elementLocs(:,1) = elementLocs(:,1) + 0.1*cos(2*pi*elementLocs(:,1) / (ymax-ymin));


k = zeros(1, 1, length(lambda));
k(1,1,:) = 2*pi ./ lambda;

fieldsTemp = get_field_patterns(lambda, 1, elementLocs, ones(numElems,1,length(lambda)), apertureNormal, xCoords, yCoords, 0);

[~, lensPlaneXCoordInd] = min(abs(xCoords - lensLocX));
lensPlaneFields = fieldsTemp(:, lensPlaneXCoordInd, :);
lensElemLocs = [transpose(repelem(xCoords(lensPlaneXCoordInd), length(yCoords))) transpose(yCoords) zeros(length(yCoords), 1)];
lensDownsamplingFactor = 2;
lensElemLocs = lensElemLocs(1:lensDownsamplingFactor:end,:);

lensPhaseShift = exp(+j*k/(2*focalLen) .* (lensElemLocs(:,2).^2));
lensOutputFields = fieldsTemp(1:lensDownsamplingFactor:end,lensPlaneXCoordInd,:) .* lensPhaseShift;

fieldsFromLens = get_field_patterns(lambda, 1, lensElemLocs, lensOutputFields, apertureNormal, xCoords, yCoords, 0);

fields = fieldsTemp;
fields(:, xCoords >= lensLocX, :) = 0;
fields(:, xCoords >= lensLocX, :) = fieldsFromLens(:, xCoords >= lensLocX, :);

synthField = fields(:,:,1) .* conj(fields(:,:,2));
synthField = sqrt(abs(synthField)) .* exp(j*angle(synthField));

% imagesc(xCoords*1000, yCoords*1000, angle(synthField));
% imagesc(xCoords*1000, yCoords*1000, abs(synthField));
% imagesc(xCoords*1000, yCoords*1000, angle(fields(:,:,1)));
imagesc(xCoords*1000, yCoords*1000, abs(fields(:,:,1)));
colorbar;
caxis([0 10000]);








%% Functions
function [g, xGrid, yGrid] = get_field_patterns(lambda, n_medium, elementLocs, inputVector, apertureNormal, xCoords, yCoords, zCut)
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

	apertureNorm = apertureNormal / sqrt(sum(apertureNormal .^ 2));

	[xGrid, yGrid] = meshgrid(xCoords, yCoords);
	g = zeros([size(xGrid) length(lambda)]);

	k_lambda = n_medium .* (2*pi ./ lambda);
	k_lambda = gpuArray(reshape(k_lambda, [1 1 1 length(k_lambda)]));
	
	dx = gpuArray(xGrid - reshape(elementLocs(:,1), [1 1 size(elementLocs, 1)]));
	dy = gpuArray(yGrid - reshape(elementLocs(:,2), [1 1 size(elementLocs, 1)]));
	dz = gpuArray(zCut - reshape(elementLocs(:,3), [1 1 size(elementLocs, 1)]));
	R = sqrt((dx.^2) + (dy.^2) + (dz.^2));
	obliquity = ((dx * apertureNorm(1)) + (dy * apertureNorm(2)) + (dz * apertureNorm(3))) ./ R;

	clear("dx", "dy", "dz");
	wait(gpuDevice);

	for m = 1:length(k_lambda)
		focusingVecTemp = gpuArray(reshape(inputVector(:,1,m), [1 1 size(inputVector, 1) 1]));
		g(:,:,m) = sum(focusingVecTemp .* (exp(-j*k_lambda(1,1,1,m).*R) ./ R) .* obliquity, 3);
	end
end