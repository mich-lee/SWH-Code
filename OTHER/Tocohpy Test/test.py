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

#Define output target image
s2=( int( s[0]/2)  ,int( s[1]/2 ) )
target_image = torch.tensor( np.array(Image.fromarray(imageio.imread('imageio:camera.png')).resize(s)), dtype =torch.cfloat ).to(device)

#define the incident field as a delta function 
field_in = torch.zeros(s, dtype = torch.cfloat, requires_grad = False).to(device)
field_in[ int(s[0]/2) - 3 : int(s[0]/2) + 3 , int(s[1]/2) - 3 : int(s[1]/2) + 3 ] = 1

# field_in = target_image


plt.imshow(target_image.abs().cpu())
plt.title('Target output image')




#### We now will build an optical system using an ordered dict data structure ####
Optical_Path_3 = OrderedDict()    

# #Display random pattern and prop light from the rear focal plane of a lens to the Fourier plane
Optical_Path_3['FT0'] =   comp.FT_Lens_NC(f= focal_length, wavelength= lamb, dx = dx , N = s[0] , device = device   )

#At the Fourier plane we apply a phase SLM. The initial pattern is a random but we 
#will learn the phase delay later
Optical_Path_3['S0'] = comp.SLM( phase_delay = 10* torch.rand(s).to(device) ,  device =device )
# Optical_Path_3['S0'] = comp.SLM( phase_delay = 0 * torch.rand(s).to(device) ,  device =device )

# #Next we propagate out of the fourier plane, through a second lens and to the detector (our DX changed using a NC method)
# Optical_Path_3['P0'] = prop.ASM_Prop( wavelength = lamb,dx = Optical_Path_3['FT0'].dx_new, distance = 1*detector_distance , N=s[0] , H=None, device = device)
Optical_Path_3['P0'] = tocohpyMods.ASM_Prop_With_Backprop( wavelength = lamb,dx = Optical_Path_3['FT0'].dx_new, distance = 1*detector_distance , N=s[0] , H=None, device = device)

#Once the Optical Path is defined we can combine it into a single network. 
diffraction_model=torch.nn.Sequential( Optical_Path_3 ).to(device)


# Adapted from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
stageOutputs = {}
def get_stage_io(name):
	def hook(model, input, output):
		stageOutputs[name] = output.detach()
	return hook
diffraction_model.FT0.register_forward_hook(get_stage_io('FT0'))
diffraction_model.S0.register_forward_hook(get_stage_io('S0'))
diffraction_model.P0.register_forward_hook(get_stage_io('P0'))


modelOutput = diffraction_model(field_in).data

# p0_inverse = prop.ASM_Prop( wavelength = lamb,dx = Optical_Path_3['FT0'].dx_new, distance = -Optical_Path_3['P0'].distance, N=s[0] , H=None, device = device)
# backpropRes = p0_inverse.forward(modelOutput)
backpropRes = diffraction_model.P0.invertForwardPropagation(diffraction_model.P0.forwardPropagate(stageOutputs['S0'], dataIsPrePadded = False, doUnpadding = False), dataIsPrePadded = True, doUnpadding = True)


# inv.inverseObject.calculate_inverse_asm_prop(Optical_Path_3['P0'], s[0], device=device)


#View the results 
# f,ax_arr = plt.subplots(1,2,figsize=(10,5))
# ax_arr[0].imshow((field_in[:,:]).abs().cpu().detach())
# ax_arr[1].imshow(modelOutput.abs().cpu().detach()) 
# ax_arr[0].set_title('Input field intensity')
# ax_arr[1].set_title('Output intensity of 4F system with aperture')
# ax_arr[0].axis('off')
# ax_arr[1].axis('off')

f,ax_arr = plt.subplots(2,3,figsize=(10,5))
ax_arr[0,0].imshow((field_in[:,:]).abs().cpu().detach())
ax_arr[0,1].imshow((stageOutputs['FT0'][:,:]).abs().cpu().detach())
ax_arr[0,2].imshow((stageOutputs['S0'][:,:]).abs().cpu().detach())
ax_arr[1,0].imshow((stageOutputs['P0'][:,:]).abs().cpu().detach())
ax_arr[1,1].imshow((backpropRes[:,:]).abs().cpu().detach())
ax_arr[1,2].imshow((modelOutput[:,:]).abs().cpu().detach())

ax_arr[0,0].set_title('Input field intensity')
ax_arr[0,1].set_title('FT0 output')
ax_arr[0,2].set_title('SLM output')
ax_arr[1,0].set_title('P0 output')
ax_arr[1,1].set_title('Undoing propagation P0')
ax_arr[1,2].set_title('Model output')

# ax_arr[0,2].set_xlim([370,429])
# ax_arr[0,2].set_ylim([370,429])
# ax_arr[1,1].set_xlim([370,429])
# ax_arr[1,1].set_ylim([370,429])

plt.show()

