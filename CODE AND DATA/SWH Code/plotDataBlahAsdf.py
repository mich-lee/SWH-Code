from json import loads
from logging import exception
from turtle import end_fill
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

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Enumerators import *

import holotorch_addons.HolotorchPatches as HolotorchPatches
from holotorch_addons.HelperFunctions import generateGrid, get_field_slice
from holotorch_addons.Mesh_Functions import createMeshFromGridsAndDepths
from holotorch_addons.Save_Load_Functions import loadSensorData

################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")


if False:
	depths = torch.load("../RESULTS/northwestern_synth_9-19-2022/groundTruthDepthData.pt")
	plt.clf()
	plt.imshow(1000*depths['depths'].cpu(), extent=[-5, 5, -5, 5])
	plt.xlabel('Width Dimension (mm)', {'fontsize':16})
	plt.ylabel('Height Dimension (mm)', {'fontsize':16})
	plt.title('Ground Truth Depthmap', {'fontsize':16,'fontweight':'bold'})
	plt.colorbar().set_label(label='Depth (mm)',size=16)
	plt.clim([0, 1.2])

elif True:
	fields = torch.load("../RESULTS/northwestern_synth_9-19-2022/Recovered Fields/SensorPlaneField_DATA_2022-9-18_07h06m11s.pt")
	fields.data = fields.data.to(device)
	fields.wavelengths = fields.wavelengths.to(device)
	fields.spacing = fields.spacing.to(device)

	i = 1
	j = 0
	l1 = float(fields.wavelengths.data_tensor[i])
	l2 = float(fields.wavelengths.data_tensor[j])
	wavelengthTemp = l1 * l2 / abs(l1 - l2)
	dataTemp = fields.data[0,0,0,i,:,:] * fields.data[0,0,0,j,:,:].conj()
	dataTemp = dataTemp[None,None,None,None,:,:]

	spacingTemp = float(fields.spacing.data_tensor[0,0,0])

	E_temp = ElectricField(data=dataTemp, wavelengths=wavelengthTemp, spacing=spacingTemp)
	# E_temp.data = E_temp.data.to(device)	# Not needed as the data in 'fields' should already be on the correct device
	E_temp.wavelengths = E_temp.wavelengths.to(device)
	E_temp.spacing = E_temp.spacing.to(device)


	plt.clf()
	plt.suptitle('Wavelengths: $\lambda_1 = ' + str(round(l1*(1e9), 2)) + 'nm$, $\lambda_2 = ' + str(round(l2*(1e9), 2)) + 'nm$ - ' + \
					'Synthetic Wavelength: $\Lambda = ' + str(round(wavelengthTemp*1000, 2)) + 'mm$')

	plt.subplot(2,3,1)
	get_field_slice(fields, batch_inds_range=0, time_inds_range=0, pupil_inds_range=0, channel_inds_range=i, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Recovered $u_{sensor}$ Field (Magnitude)\n' + '$\lambda_1 = ' + str(round(l1*(1e9), 2)) + 'nm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	plt.subplot(2,3,4)
	get_field_slice(fields, batch_inds_range=0, time_inds_range=0, pupil_inds_range=0, channel_inds_range=i, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Recovered $u_{sensor}$ Field (Phase) - $\lambda_1 = ' + str(round(l1*(1e9), 2)) + 'nm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	plt.subplot(2,3,2)
	get_field_slice(fields, batch_inds_range=0, time_inds_range=0, pupil_inds_range=0, channel_inds_range=j, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Recovered $u_{sensor}$ Field (Magnitude)\n' + '$\lambda_2 = ' + str(round(l2*(1e9), 2)) + 'nm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	plt.subplot(2,3,5)
	get_field_slice(fields, batch_inds_range=0, time_inds_range=0, pupil_inds_range=0, channel_inds_range=j, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Recovered $u_{sensor}$ Field (Phase) - $\lambda_2 = ' + str(round(l2*(1e9), 2)) + 'nm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	plt.subplot(2,3,3)
	E_temp.visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Synthetic Field (Magnitude)\n' + '$\Lambda = ' + str(round(wavelengthTemp*(1e3), 2)) + 'mm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	plt.subplot(2,3,6)
	E_temp.visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.xlabel('Width Dimension (mm)')
	plt.ylabel('Height Dimension (mm)')
	plt.title('Synthetic Field (Phase) - $\Lambda = ' + str(round(wavelengthTemp*(1e3), 2)) + 'mm$')
	plt.xlim(-1,1)
	plt.ylim(-1,1)

	# axes = plt.gcf().get_axes()
	# for i in range(len(axes)):
	# 	if len(axes[i].get_images()) != 0:
	# 		axes[i].get_images()[0].set_interpolation('nearest')

elif False:
	dataFolderPath = pathlib.Path('../DATA/Synthetic Data/DATA_2022-9-18_07h06m11s')
	tempSlmDataFolderPath = pathlib.Path('../DATA/.temp_slm')
	channel_num = 0		# Selects the wavelength
	sensorDataPlotDownsamplingFactor = 8

	sensorData = loadSensorData(dataFolderPath=dataFolderPath, device=device)

	slmModel =	HolotorchPatches.SLM_PhaseOnly_Patched.create_slm(	height          		= 1,
																	width           		= 1,
																	feature_size    		= 1,
																	init_type       		= ENUM_SLM_INIT.RANDOM, # This should not matter as we are loading SLM data
																	slm_directory			= tempSlmDataFolderPath.resolve(),
																	static_slm				= True,
																	static_slm_data_path	= dataFolderPath / 'SLM_Data',
																	store_on_gpu			= True,
																	device					= device
																)

	n_slm_batches = slmModel.n_slm_batches
	n_batch = slmModel.batch_tensor_dimension.batch
	n_wavelengths = slmModel.batch_tensor_dimension.channel

	if True:
		plt.figure(1)
		plt.clf()
		for j in range(1):
			sensorDataTemp = sensorData[j].data[:,0,0,channel_num,:,:]
			slmModel.load_single_slm(batch_idx=j)
			slmDataTemp = slmModel()
			for i in range(n_batch):
				sensorDataTempPlot = sensorDataTemp[i,0::sensorDataPlotDownsamplingFactor,0::sensorDataPlotDownsamplingFactor].cpu()
				plt.subplot(2, n_batch, i + 1)
				plt.imshow(slmDataTemp[i,0,channel_num,:,:].angle().detach().cpu(), cmap=plt.colormaps['hsv'],
							extent=[-1920*6.4e-6*1000/2, 1920*6.4e-6*1000/2, -1080*6.4e-6*1000/2, 1080*6.4e-6*1000/2])
				plt.title('SLM Pattern #' + str(i+1))
				plt.xlabel('Horizontal Position (mm)')
				plt.ylabel('Vertical Position (mm)')
				plt.subplot(2, n_batch, n_batch + i + 1)
				plt.imshow(sensorDataTempPlot, cmap=plt.colormaps['turbo'],
							extent=[-4024*1.85e-6*1000/2, 4024*1.85e-6*1000/2, -3036*1.85e-6*1000/2, 3036*1.85e-6*1000/2])
				plt.title('Sensor Image #' + str(i+1))
				plt.xlabel('Horizontal Position (mm)')
				plt.ylabel('Vertical Position (mm)')

	elif False:
		# plt.figure(1)
		# plt.clf()
		# for j in range(n_slm_batches):
		# 	slmModel.load_single_slm(batch_idx=j)
		# 	slmDataTemp = slmModel()
		# 	for i in range(n_batch):
		# 		plt.subplot(n_batch, n_slm_batches, i*n_slm_batches + j + 1)
		# 		plt.imshow(slmDataTemp[i,0,channel_num,:,:].angle().detach().cpu())

		fig = plt.figure()
		plt.clf()
		# plt.imshow(sensorData[0].data[:,0,0,channel_num,:,:][0,0::sensorDataPlotDownsamplingFactor,0::sensorDataPlotDownsamplingFactor].cpu(), cmap=plt.colormaps['binary'])
		for j in range(3): # for j in range(n_slm_batches):
			sensorDataTemp = sensorData[j].data[:,0,0,channel_num,:,:]
			slmModel.load_single_slm(batch_idx=j+2)
			slmDataTemp = slmModel()
			for i in range(2): # for i in range(n_batch):
				# plt.subplot(n_batch, n_slm_batches, i*n_slm_batches + j + 1)
				ax = plt.subplot(2, 3, i*3 + j + 1)
				ax_pos = ax.get_position('original')
				sensorDataTempPlot = sensorDataTemp[i,0::sensorDataPlotDownsamplingFactor,0::sensorDataPlotDownsamplingFactor].cpu()
				plt.imshow(sensorDataTempPlot,
							cmap=plt.colormaps['turbo'], extent=[-4024*1.85e-6*1000/2, 4024*1.85e-6*1000/2, -3036*1.85e-6*1000/2, 3036*1.85e-6*1000/2])
				ax.xaxis.set_visible(False)
				ax.yaxis.set_visible(False)

				ax_w = ax_pos.xmax - ax_pos.xmin
				ax_h = ax_pos.ymax - ax_pos.ymin
				ax2_w = 0.3 * ax_w
				ax2_h = 0.3 * ax_h
				ax2_padw = 0.025 * ax_w
				ax2_padh = 0.025 * ax_h
				ax2 = fig.add_axes([0, 0, 0.05, 0.05])
				plt.imshow(slmDataTemp[i+2,0,channel_num,:,:].angle().detach().cpu(), cmap=plt.colormaps['hsv'],
							extent=[-1920*6.4e-6*1000/2, 1920*6.4e-6*1000/2, -1080*6.4e-6*1000/2, 1080*6.4e-6*1000/2])
				plt.title('SLM Pattern', {'color': 'white'})
				ax2.xaxis.set_visible(False)
				ax2.yaxis.set_visible(False)
				ax2.set_position([ax_pos.xmax - ax2_w - ax2_padw, ax_pos.ymin + ax2_padh, ax2_w, ax2_h], which='original')

		plt.suptitle('Sensor Images (Intensity)\n' + r'$\lambda$ = ' + str(round(float(sensorData[0].wavelengths.data_tensor[channel_num]*1e9), 2)) + r'nm', fontsize=24, fontweight='bold')

elif False:
	objPlaneField = torch.load("../RESULTS/northwestern_synth_9-19-2022/synth_data_object_plane_input_field.pt")
	channelNum = 0

	tempField = get_field_slice(objPlaneField, channel_inds_range=channelNum, field_data_tensor_dimension=Dimensions.BTPCHW)

	plt.figure()
	plt.subplot(1,2,1)
	tempField.visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.title('Object Plane Field (Magnitude)', {'fontsize':32,'fontweight':'bold'})
	plt.subplot(1,2,2)
	tempField.visualize(plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.title('Object Plane Field (Phase)', {'fontsize':32,'fontweight':'bold'})


1 == 1




# Run this while at a breakpoint inside Synthetic_Data_Generator.py:
# fig = plt.figure(1)
# plt.clf()
# ax = plt.subplot(1,1,1)
# ax_pos = ax.get_position('original')
# plotField = get_field_slice(self.slmInputPlaneField, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).data.squeeze().cpu().detach()
# plt.imshow(plotField[270:810,480:1440].abs(), extent=[-1920*6.4e-6*1000/4, 1920*6.4e-6*1000/4, -1080*6.4e-6*1000/4, 1080*6.4e-6*1000/4])
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # ax.xaxis.set_visible(False)
# # ax.yaxis.set_visible(False)
# plt.xlabel('Horizontal Position (mm)', {'fontsize': 24})
# plt.ylabel('Vertical Position (mm)', {'fontsize': 24})
# plt.title('SLM Input Plane Field (Magnitude)', {'fontsize': 38, 'fontweight': 'bold'})
# plt.text(-3, -1.65, 'Zoomed-In View (150%)', {'color': 'white', 'fontsize': 36, 'fontweight': 'bold'})
# plt.set_cmap('turbo')
# ax_w = ax_pos.xmax - ax_pos.xmin
# ax_h = ax_pos.ymax - ax_pos.ymin
# ax2_w = 0.25 * ax_w
# ax2_h = 0.25 * ax_h
# ax2_padw = 0.06 * ax_w
# ax2_padh = 0.025 * ax_h
# ax2 = fig.add_axes([0, 0, 0.05, 0.05])
# ax2.set_position([ax_pos.xmax - ax2_w - ax2_padw, ax_pos.ymin + ax2_padh, ax2_w, ax2_h], which='original')
# ax2.spines['top'].set_color('white')
# ax2.spines['bottom'].set_color('white')
# ax2.spines['left'].set_color('white')
# ax2.spines['right'].set_color('white')
# plt.imshow(plotField.abs(), extent=[-1920*6.4e-6*1000/2, 1920*6.4e-6*1000/2, -1080*6.4e-6*1000/2, 1080*6.4e-6*1000/2])
# plt.title('Full View', {'color': 'white', 'fontsize': 36, 'fontweight': 'bold'})
# ax2.xaxis.set_visible(False)
# ax2.yaxis.set_visible(False)