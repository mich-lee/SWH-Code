from logging import exception
import numpy as np
import sys
import torch
import torchvision
import matplotlib.pyplot as plt

import pathlib
import glob
import copy
# import datetime

from stl import mesh

import kornia

from numpy import asarray
import gc	# For garbage collection/freeing up memory

# Image wranglers
import imageio
from PIL import Image

import warnings

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")
sys.path.append("holotorch_addons/")

from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
import holotorch.utils.Visualization_Helper as VH

from holotorch_addons.HelperFunctions import generateGrid
from holotorch_addons.Mesh_Functions import createMeshFromGridsAndDepths

################################################################################################################################









################################################################################################################################
# filepath = pathlib.Path("../DATA/Recovered Depth Maps/RecoveredDepthMap_DATA_2022-9-11_16h18m11s.pt")
# filepath = pathlib.Path("../DATA/Recovered Depth Maps/DepthMap_1_0_2022-9-18_18h14m/34s.pt")
filepath = pathlib.Path("../RESULTS/northwestern_synth_9-19-2022/Recovered Depth Maps/DepthMap_1_0_2022-9-18_18h14m34s.pt")


depthMapData = torch.load(filepath)


depthGridOrig = depthMapData['depth'].detach().squeeze()	# detach() is done in case the saved data had requires_grad=True
xGridOrig = depthMapData['xGrid'].detach()
yGridOrig = depthMapData['yGrid'].detach()

# This print statement is to serve as a reminder that there is more data (e.g. wavelengths, synthetic wavelength, etc.)
# saved in depthMapData.
print(depthMapData)

# Specify how the image was magnified
M = -75*mm / (75*mm - 500*mm)

# 'nanDepthValue' decides what values depths of NaN will take on.
# NaN values are given if the magnitude of the field at a certain point is sufficiently small (e.g. there is nothing solid at that point)
nanDepthValue = 0

# Controls how many datapoints are transferred from the loaded depth data to the actual mesh
# This helps keep meshes from becoming too large
# A value of M for this will scale the number of vertices down to 1/(M^2) times the original number of vertices
subsamplingMagnitude = 4





nHeight = xGridOrig.shape[0]
nWidth = xGridOrig.shape[1]

dx = xGridOrig[1,0] - xGridOrig[0,0]
dy = yGridOrig[0,1] - yGridOrig[0,0]

depthShiftAmount = -depthGridOrig[~depthGridOrig.isnan()].min()

# Clean up depth data
outOfRangeValue = depthGridOrig[~depthGridOrig.isnan()].min() - 1
depthGrid = depthGridOrig.clone()
depthGrid[depthGridOrig.isnan()] = outOfRangeValue
depthGrid = kornia.filters.median_blur(depthGrid[None,None,:,:], kernel_size=(5,5)).squeeze()
trueNaNs = (depthGrid == outOfRangeValue)
depthGrid[trueNaNs] = 0
blurSigma = float((7.5e-6) / ((dx + dy)/2))
depthGrid = torchvision.transforms.functional.gaussian_blur(depthGrid[None,:,:], 17, blurSigma)[0,:,:]
depthGrid = depthGrid + (depthShiftAmount * (~trueNaNs))
depthGrid[trueNaNs] = nanDepthValue

depthGridRaw = depthGridOrig.clone()
depthGridRaw += depthShiftAmount
depthGridRaw[depthGridOrig.isnan()] = nanDepthValue




# Plotting to help visualize the image plane data
imshowPlotExtent = (yGridOrig.min() - dy/2, yGridOrig.max() + dy/2, xGridOrig.min() - dx/2, xGridOrig.max() + dx/2)
imshowPlotExtent = torch.tensor(imshowPlotExtent)

plt.figure(figsize=(10,10))
imshowPlotExtent = imshowPlotExtent * np.abs(1/M)	# Scale back to object plane size
plt.clf()
plt.suptitle('Wavelengths: $\lambda_1 = 854.43nm, \lambda_2 = 854.31nm$\n' + \
				'Synthetic Wavelength: $\Lambda = 6.08mm$', fontsize=28, fontweight='bold')
# plt.subplot(2,2,1)
plt.subplot(1,2,1)
im1 = plt.imshow((depthGridRaw*1000).cpu(), extent=imshowPlotExtent*1000)
plt.xlabel('Width Dimension (mm)', {'fontsize': 20})
plt.ylabel('Height Dimension (mm)', {'fontsize': 20})
plt.title('Recovered Depth Map (Initial)', {'fontsize': 24, 'fontweight': 'bold'})
# plt.xlim(-0.0008,0.0008)
# plt.ylim(-0.0008,0.0008)
plt.xlim(0.0008*np.abs(1/M)*1000,-0.0008*1000*np.abs(1/M))
plt.ylim(0.0008*np.abs(1/M)*1000,-0.0008*1000*np.abs(1/M))
VH.add_colorbar(im1).set_label(label='Depth (mm)',size=20)
plt.clim(0, 1.2)
# plt.subplot(2,2,2)
plt.subplot(1,2,2)
im2 = plt.imshow((depthGrid*1000).cpu(), extent=imshowPlotExtent*1000)
plt.xlabel('Width Dimension (mm)', {'fontsize': 20})
plt.ylabel('Height Dimension (mm)', {'fontsize': 20})
plt.title('Recovered Depth Map (Processed)', {'fontsize': 24, 'fontweight': 'bold'})
# plt.xlim(0.0008,-0.0008)
# plt.ylim(0.0008,-0.0008)
plt.xlim(0.0008*1000*np.abs(1/M),-0.0008*1000*np.abs(1/M))
plt.ylim(0.0008*1000*np.abs(1/M),-0.0008*1000*np.abs(1/M))
VH.add_colorbar(im2).set_label(label='Depth (mm)',size=20)
plt.clim(0, 1.2)

# asdf = torch.load('../RESULTS/northwestern_synth_9-19-2022/groundTruthDepthData.pt')
# imshowPlotExtent2 = torch.tensor([asdf['gridX'].min() - (2.5e-6)/2, asdf['gridX'].max() + (2.5e-6)/2, asdf['gridY'].min() - (2.5e-6)/2, asdf['gridY'].max() + (2.5e-6)/2])
# plt.subplot(2,1,2)
# plt.imshow(asdf['depths'].cpu(), extent=imshowPlotExtent2*1000)
# plt.xlabel('Width Dimension (mm)')
# plt.ylabel('Height Dimension (mm)')
# plt.title('"Ground Truth" Depths')
# plt.colorbar()
# plt.clim(0, 0.0012)



# Scale to remap back to the object plane transverse coordinates
#	(I.e. undo the magnification)
xGridMesh = (1/M) * xGridOrig
yGridMesh = (1/M) * yGridOrig

depthMeshRaw = createMeshFromGridsAndDepths(xGridMesh, yGridMesh, depthGridRaw, subsamplingMagnitude=subsamplingMagnitude)
depthMeshProcessed = createMeshFromGridsAndDepths(xGridMesh, yGridMesh, depthGrid, subsamplingMagnitude=subsamplingMagnitude)

while True:
	depthDataFilename = filepath.stem
	resp = input("Save depth map data? (y/n): ")
	if (resp == 'y'):
		depthMeshRaw.save('STL_RAW_' + depthDataFilename + '.stl')
		depthMeshProcessed.save('STL_Processed_' + depthDataFilename + '.stl')
		print("Saved data.")
		print("Exiting...")
		break
	elif (resp == 'n'):
		print("Exiting...")
		break
	else:
		print("Invalid input.")