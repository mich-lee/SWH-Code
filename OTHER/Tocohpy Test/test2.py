#Basic Packages
import sys
import torch
import numpy as np

# import time
import matplotlib.pyplot as plt

# Image wranglers
import imageio
from collections import OrderedDict
from PIL import Image

#Tocohpy functions 
sys.path.append('Tocohpy/')
import Tocohpy.Optical_Components as comp
import Tocohpy.Optical_Propagators as prop
import Tocohpy.Helper_Functions as HF

from Helper_Functions import *

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
field_in = torch.zeros(s, dtype = torch.cfloat, requires_grad = False).to(device)
field_in = torch.exp(1j*2*np.pi*(target_image.abs()/256)*0.25) * (target_image.abs() > 0)
# field_in = torch.exp(1j*2*np.pi*(target_image.abs()/256)*0.25)


# plt.imshow(target_image.abs().cpu())
# plt.title('Target output image')


# Random phase patterns
N_Patterns = 16
phasePatterns = torch.zeros(1, N_Patterns, s[0], s[1])
for i in range(1,N_Patterns):
	phasePatterns[0,i,:,:] = 10*torch.rand(s).to(device)



#### We now will build an optical system using an ordered dict data structure ####
Optical_Path_3 = OrderedDict()

#Apply a phase SLM
Optical_Path_3['S0'] = comp.SLM( phase_delay = phasePatterns,  device =device )
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

# backpropRes = diffraction_model.P0.invertForwardPropagation(stageOutputs['P0'], dataIsPrePadded = False, doUnpadding = True)

# forwardSlmPhaseMtx = torch.exp(1j*phasePatterns)
inverseSlmPhaseMtx = torch.exp(-1j*phasePatterns).to(device=device)
uslm = (torch.sum(inverseSlmPhaseMtx * diffraction_model.P0.invertForwardPropagation(intensityMeasSqrt, dataIsPrePadded = False, doUnpadding = True), 1))[0,:,:] / N_Patterns
GS_epsilon = 1e-7
numElems = s[0]*s[1]*N_Patterns
numIter = 0
smallestError = np.Infinity
bestUslm = []
uslmNIter = []
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
	if (numIter == 3):
		uslmNIter = uslm
uslm = bestUslm
# uslm[uslm.abs() < 0.1] = 0




#View the results
field_in_avg_phase = torch.sum(field_in[:,:].angle().cpu().detach(), (0,1)) / (field_in.shape[0] * field_in.shape[1])
uslm_avg_phase = torch.sum(uslm[:,:].angle().cpu().detach(), (0,1)) / (uslm.shape[0] * uslm.shape[1])

f,ax_arr = plt.subplots(2,3,figsize=(10,5))
im_arr = [0,0,0,0,0,0]
im_arr[0] = ax_arr[0,0].imshow(torch.sum(modelOutputInitial[:,:], 1)[0,:,:].abs().cpu().detach())
im_arr[1] = ax_arr[0,1].imshow((field_in[:,:]).abs().cpu().detach())
im_arr[2] = ax_arr[0,2].imshow((field_in[:,:]).angle().cpu().detach())
# im_arr[2] = ax_arr[0,2].imshow((field_in[:,:] - field_in_avg_phase).angle().cpu().detach())
im_arr[3] = ax_arr[1,0].imshow((uslmNIter[:,:]).angle().cpu().detach())
im_arr[4] = ax_arr[1,1].imshow((uslm[:,:]).abs().cpu().detach())
im_arr[5] = ax_arr[1,2].imshow((uslm[:,:]).angle().cpu().detach())
# im_arr[5] = ax_arr[1,2].imshow((uslm[:,:] - uslm_avg_phase).angle().cpu().detach())

plt.colorbar(im_arr[0], ax=ax_arr[0,0])
plt.colorbar(im_arr[1], ax=ax_arr[0,1])
plt.colorbar(im_arr[2], ax=ax_arr[0,2])
plt.colorbar(im_arr[3], ax=ax_arr[1,0])
plt.colorbar(im_arr[4], ax=ax_arr[1,1])
plt.colorbar(im_arr[5], ax=ax_arr[1,2])

ax_arr[0,0].set_title('Average √I for all SLM patterns at sensor plane')
ax_arr[0,1].set_title('Field at SLM Input (Magnitude)')
ax_arr[0,2].set_title('Field at SLM Input (Phase)')
ax_arr[1,0].set_title('U_{slm} phase after three iterations of GS')
ax_arr[1,1].set_title('Recovered U_{slm} (Magnitude)')
ax_arr[1,2].set_title('Recovered U_{slm} (Phase)')

plt.show()

qwerty = 123