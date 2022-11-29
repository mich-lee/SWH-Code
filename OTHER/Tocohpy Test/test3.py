#Basic Packages
from cmath import sqrt
import sys
import torch
import numpy as np

import time
import matplotlib.pyplot as plt

# Image wranglers
import imageio
from collections import OrderedDict
from PIL import Image
from InputDummyLayer import InputLDummyLayer

#Tocohpy functions 
sys.path.append('Tocohpy/')
import Tocohpy.Optical_Components as comp
import Tocohpy.Optical_Propagators as prop
import Tocohpy.Helper_Functions as HF

# from Helper_Functions import *

# import Custom as mods
import Tocohpy_Mods as tocohpyMods

##GPU info for pytorch##
gpu_no = 0
use_cuda = True
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")


plt.style.use('dark_background')
""" This Notebook is an example notebook for Optical NN learning with Tocohpy - Lionel Fiske  """
#Last update: 9/15/2021


#Setting uo problem parameters 

# Settings
phaseRecoveryAlgorithmSelector = 1		# 0 for Gerchberg-Saxton, 1 for Torch optimizer

# units
mm = 1e-3
nm = 1e-9


## Setting Problem Parameters ##

#Masking the initial field 
s= (800,800)      # size of simulation in pixels
sim_size = 5*mm     # size of wavefront 

#coordinates 
x_c = torch.linspace(-sim_size/2,sim_size/2, s[0] )  #XCoordinates 
y_c = torch.linspace(-sim_size/2,sim_size/2, s[1] )  #YCoordinates 
[X,Y] = torch.meshgrid(x_c,y_c)    #Generate Coordinate Mesh
R2= X**2 + Y**2                    #Radial coordinates
dx = x_c[1] - x_c[0]                #grid spacing


# Wavelength
lamb  = 500.0* nm

#Lens focal length
focal_length = 30* mm 

#Detector distance
detector_distance = 20* mm

#Define image
s2=( int( s[0]/2)  ,int( s[1]/2 ) )
# target_image = torch.tensor( np.array(Image.fromarray(imageio.imread('imageio:camera.png')).resize(s)), dtype =torch.cfloat ).to(device)

#define the incident field as a delta function 
field_in = torch.zeros(s, dtype = torch.cfloat, requires_grad = False).to(device)
field_in[ int(s[0]/2) - 3 : int(s[0]/2) + 3 , int(s[1]/2) - 3 : int(s[1]/2) + 3 ] = 1
# field_in[ int(s[0]/2) - 10 : int(s[0]/2) + 10 , int(s[1]/2) - 10 : int(s[1]/2) + 10 ] = 1

target_image = torch.tensor( np.array(Image.fromarray(imageio.imread('testImage3.png')).resize(s)), dtype =torch.cfloat ).to(device)

# field_in = torch.zeros(s, dtype = torch.cfloat, requires_grad = False).to(device)
# field_in = torch.exp(1j*2*np.pi*(target_image.abs()/256)*0.25) * (target_image.abs() > 0)
# field_in = torch.exp(1j*2*np.pi*(target_image.abs()/256)*0.25)

tempGridX, tempGridY = np.meshgrid(list(range(s[0])), list(range(s[1])))
tempGridX = tempGridX - ((s[0] - 1) / 2)*np.ones(s)
tempGridY = tempGridY - ((s[1] - 1) / 2)*np.ones(s)
tempMagnitudeMask = np.ones(s) - np.sqrt((tempGridX**2) + (tempGridY**2)) / np.sqrt((((s[0] - 1) / 2)**2) + (((s[1] - 1) / 2)**2))
tempMagnitudeMask = 0.3*tempMagnitudeMask*np.cos((tempGridX + tempGridY) / 100)*np.cos((tempGridX - 0.7*tempGridY) / 60) + 0.6*np.ones(s)
field_in = torch.exp(1j*2*np.pi*(target_image.abs()/256)*0.4) * torch.tensor(tempMagnitudeMask).to(device)


# plt.imshow(target_image.abs().cpu())
# plt.title('Target output image')


# Random phase patterns
N_Patterns = 16
phasePatterns = torch.zeros(1, N_Patterns, s[0], s[1])
for i in range(1,N_Patterns):
	phasePatterns[0,i,:,:] = 10*torch.rand(s).to(device)



#### We now will build an optical system using an ordered dict data structure ####
Optical_Path_3 = OrderedDict()

# Making a dummy input layer
if phaseRecoveryAlgorithmSelector == 0:
	useFixedInputWeights = True
elif phaseRecoveryAlgorithmSelector == 1:
	useFixedInputWeights = False
Optical_Path_3['I0'] = InputLDummyLayer(weights = torch.ones(s), device=device, fixed_weights=useFixedInputWeights)

#Apply a phase SLM
Optical_Path_3['S0'] = comp.SLM( phase_delay = phasePatterns,  device =device, fixed_pattern=True )
# Optical_Path_3['S0'] = comp.SLM( phase_delay = 10*torch.rand(s).to(device) ,  device =device )
# Optical_Path_3['S0'] = comp.SLM( phase_delay = 0 * torch.rand(s).to(device) ,  device =device )

# Propagate through a second lens and to the detector (our DX changed using a NC method)
# Optical_Path_3['P0'] = prop.ASM_Prop( wavelength = lamb,dx = Optical_Path_3['FT0'].dx_new, distance = 1*detector_distance , N=s[0] , H=None, device = device)
Optical_Path_3['P0'] = tocohpyMods.ASM_Prop_With_Backprop( wavelength = lamb,dx = dx, distance = 1*detector_distance , N=s[0] , H=None, device = device)

#Once the Optical Path is defined we can combine it into a single network. 
diffraction_model=torch.nn.Sequential( Optical_Path_3 ).to(device)


# Adapted from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
# stageOutputs = {}
# def get_stage_io(name):
# 	def hook(model, input, output):
# 		stageOutputs[name] = output.detach()
# 	return hook
# diffraction_model.S0.register_forward_hook(get_stage_io('S0'))
# diffraction_model.P0.register_forward_hook(get_stage_io('P0'))


modelOutputInitial = diffraction_model(field_in).data
intensityMeas = modelOutputInitial.abs() ** 2
intensityMeasSqrt = torch.sqrt(intensityMeas)


# forwardSlmPhaseMtx = torch.exp(1j*phasePatterns)
inverseSlmPhaseMtx = torch.exp(-1j*phasePatterns).to(device=device)
numElems = s[0]*s[1]*N_Patterns

if phaseRecoveryAlgorithmSelector == 0:
	# Performing Gerchberg-Saxton	
	GS_epsilon = 1e-7
	numIter = 0
	smallestError = np.Infinity
	uslm = (torch.sum(inverseSlmPhaseMtx * diffraction_model.P0.invertForwardPropagation(intensityMeasSqrt, dataIsPrePadded = False, doUnpadding = True), 1))[0,:,:] / N_Patterns
	bestUslm = []
	while (numIter <= 30):
		numIter = numIter + 1
		y = diffraction_model(uslm).data
		errTemp = abs(torch.sum(y.abs() - intensityMeasSqrt, (0,1,2,3))) / numElems
		if (numIter % 10 == 0):
			print("Iteration " + str(numIter) + ": " + str(errTemp))
			sys.stdout.flush()
		if (errTemp < smallestError):
			smallestError = errTemp
			bestUslm = uslm
		if (errTemp < GS_epsilon):
			break
		yc = intensityMeasSqrt * torch.exp(1j*torch.angle(y))
		uslmTemp = inverseSlmPhaseMtx * diffraction_model.P0.invertForwardPropagation(yc, dataIsPrePadded = False, doUnpadding = True)
		uslm = torch.sum(uslmTemp, 1)[0,:,:] / N_Patterns
	uslm = bestUslm
elif phaseRecoveryAlgorithmSelector == 1:
	# Using PyTorch optimizer to perform phase recovery
	optimizer = torch.optim.Adam(diffraction_model.parameters() , lr=0.5)
	optimEpsilon = 1e-6
	smallestLoss = np.Infinity
	bestUslm = []
		# Saving this just in case:
			# initWeights = Optical_Path_3['I0'].weights
			# initWeights = torch.nn.Parameter(field_in.to(device).type(Optical_Path_3['I0'].dtype),requires_grad=True)
			# initWeights = torch.nn.Parameter(uslmInit.to(device).type(Optical_Path_3['I0'].dtype),requires_grad=True)
			# Optical_Path_3['I0'].weights = initWeights
			# Optical_Path_3['I0'] = InputLDummyLayer(weights = uslmInit, device=device, fixed_weights=False)
	optFieldIn = torch.ones(s).to(device)
	for t in range(500):
		#L2: Compute and print loss
		y = diffraction_model(optFieldIn)
		L_fun = torch.sum((y.abs() - intensityMeasSqrt) ** 2, (0,1,2,3)).abs() / numElems

		if (L_fun.item() < smallestLoss):
			smallestLoss = L_fun.item()
			bestUslm = Optical_Path_3['I0'].weights

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		L_fun.backward()
		optimizer.step()

		if t % 1 == 0:
			print('Iteration: ', t+1, '\t|\tCurrent Loss:' , L_fun.item(), '\t|\tSmallest Loss: ', smallestLoss)
			sys.stdout.flush()

		if (smallestLoss < optimEpsilon):
			break
	uslm = bestUslm




# Correct absolute phase for plotting comparison
tempSmallestMSE = np.Infinity
tempBestPhi = 0
tempLims = [-np.pi, np.pi]
for i in range(10):
	for phi in np.linspace(tempLims[0], tempLims[1], 181):
		tempUslm = uslm * torch.exp(1j * torch.tensor(phi))
		tempMSE = torch.sum((tempUslm - field_in) ** 2, (0,1)).abs() / numElems
		if (tempMSE < tempSmallestMSE):
			tempSmallestMSE = tempMSE
			tempBestPhi = phi
	# print(tempBestPhi, '\t', tempLims[0], '\t', tempLims[1], '\t', tempSmallestMSE.detach().cpu().numpy()*1)
	tempSpan = tempLims[1] - tempLims[0]
	tempLims = np.minimum([np.pi, np.pi], np.maximum([-np.pi, -np.pi], [tempBestPhi - (tempSpan/4), tempBestPhi + (tempSpan/4)]))
uslmAbsPhaseCorrected = uslm * torch.exp(1j * torch.tensor(tempBestPhi))




#View the results

plt.subplot(2, 3, 1)
plt.imshow((torch.sum(modelOutputInitial[:,:], 1)[0,:,:] / (uslm.shape[0] * uslm.shape[1])).abs().cpu().detach())
plt.colorbar()
plt.title('Average √I for all SLM patterns at sensor plane')

plt.subplot(2, 3, 2)
plt.imshow((uslm[:,:]).abs().cpu().detach())
plt.colorbar()
plt.title('Recovered U_{slm} (Magnitude)')

plt.subplot(2, 3, 3)
plt.imshow((field_in[:,:]).abs().cpu().detach())
plt.colorbar()
plt.title('Field at SLM Input (Magnitude)')

plt.subplot(2, 3, 4)
plt.imshow((uslm[:,:]).angle().cpu().detach())
plt.colorbar()
plt.clim(-np.pi, np.pi)
plt.title('Recovered U_{slm} (Phase)')

plt.subplot(2, 3, 5)
plt.imshow((uslmAbsPhaseCorrected[:,:].angle()).cpu().detach())
plt.colorbar()
plt.clim(-np.pi, np.pi)
plt.title('Recovered U_{slm} (Phase w/ abs. phase correction)')

plt.subplot(2, 3, 6)
plt.imshow((field_in[:,:]).angle().cpu().detach())
plt.colorbar()
plt.clim(-np.pi, np.pi)
plt.title('Field at SLM Input (Phase)')

# 	# Saving this just in case:
# 		# field_in_avg_phase = torch.sum(field_in[:,:].angle().cpu().detach(), (0,1)) / (field_in.shape[0] * field_in.shape[1])
# 		# field_in_avg_phase2 = torch.mean(field_in[field_in.abs() >= 0.01].angle())
# 		# uslm_avg_phase = torch.sum(uslm[:,:].angle().cpu().detach(), (0,1)) / (uslm.shape[0] * uslm.shape[1])
# 		# uslm_avg_phase2 = torch.mean(uslm[field_in.abs() >= 0.01].angle())

plt.show()