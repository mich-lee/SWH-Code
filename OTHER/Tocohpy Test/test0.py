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
sys.path.append('../Tocohpy')
import Optical_Components as comp
import Optical_Propagators as prop
import Helper_Functions as HF

##GPU info for pytorch##
gpu_no = 0
use_cuda = True
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")


plt.style.use('dark_background')
""" This Notebook is an example notebook for Optical NN learning with Tocohpy - Lionel Fiske  """
#Last update: 9/15/2021

## In this example a 4 f system will be modeled.
##

# units
mm = 1e-3 #meters
nm = 1e-9 #meters


### Setting Problem Parameters ###

#Masking the initial field 
s= (1024, 1024)      # size of simulation in pixels
sim_size = 5*mm      # size of wavefront / simulation fox

#coordinates 
x_c = torch.linspace(-sim_size/2,sim_size/2, s[0] )  #XCoordinates 
y_c = torch.linspace(-sim_size/2,sim_size/2, s[1] )  #YCoordinates 
[X,Y] = torch.meshgrid(x_c,y_c)    #Generate Coordinate Mesh
R2= X**2 + Y**2                    #Radial coordinates
dx = x_c[1] - x_c[0]               #grid spacing


# Wavelength
lamb  = 500.0* nm

#Lens focal length
focal_length = 25* mm 

#Define output target image

#We load an image in and apply some padding 
s2=( int( s[0]/2)  ,int( s[1]/2 ) )
field_in = torch.zeros(2 , 1, s[0], s[1], device = device, dtype = torch.cfloat)

#Here I am loading in a few images and resampling them so they are the same size as my field
resized_image = torch.tensor( np.array(Image.fromarray(imageio.imread('imageio:camera.png')).resize(s2)) )
image_noise_padding = torch.nn.functional.pad(resized_image+ 0*torch.rand(s2), (int(1/4 * s[0]),int(1/4 * s[0]),int(1/4 * s[1]),int(1/4 * s[1])), mode='constant', value=0)   

resized_image2 =torch.tensor( np.array(Image.fromarray(imageio.imread('imageio:astronaut.png')[:,:,0]).resize(s2)) )
image_noise_padding2 = torch.nn.functional.pad(resized_image2+ 0*torch.rand(s2), (int(1/4 * s[0]),int(1/4 * s[0]),int(1/4 * s[1]),int(1/4 * s[1])), mode='constant', value=0)   


## Tocohpy uses pytorch, so it can handle batch and channel dimensions.
## load each image into a different batch number
field_in[0,0,:,:] = ((1j)*image_noise_padding/255.0  ).to(device) 
field_in[1,0,:,:] = ((1j)*image_noise_padding2/255.0  ).to(device) 

####################################################

#### We now will build an optical system using an ordered dict data structure ####
Optical_Path = OrderedDict()    

# #Display an image and prop light 1 focal length to the lens. Each step gets assigned a name of your choice. 
# We adopt the convention that propagation steps begin with P, lenses with L, SLMs with S and absorption masks with A
Optical_Path['P0'] = prop.ASM_Prop( wavelength = lamb,dx = dx, distance = 1*focal_length , N=s[0] , H=None, device = device)

# #We apply a thin lens phase delay and propagate 1 more f to the fourier plane 
Optical_Path['L1'] = comp.Thin_Lens(f= 1*focal_length, wavelength=lamb, R2 = R2 , device = device )
Optical_Path['P1'] = prop.ASM_Prop( wavelength = lamb,dx = dx , distance = 1*focal_length , N=s[0] , H=None, device = device)


#At the Fourier plane we apply a low pass absorption mask. 
#We define the mask as transmitting 1 within a certain radius and transmitting 0 elsewhere
mask= torch.ones(s)
aperture_radius_sq = .25*torch.max( R2 )
mask[R2> aperture_radius_sq] = 0

#We set an absorption grating with this mask. Since we wont optimize for this variable
#we will pass a True flag to fixed_pattern
Optical_Path['A0'] = comp.Absorption_Mask( transmission = mask.clone(), fixed_pattern = True , device =device )


# #Nex we propagate out of the fourier plane, through a second lens and to the detector
Optical_Path['P2'] = prop.ASM_Prop( wavelength = lamb,dx =dx, distance = 1*focal_length , N=s[0] , padding = 0, H=None, device = device)
Optical_Path['L2'] = comp.Thin_Lens(f= 1*focal_length, wavelength=lamb, R2 = R2  , device = device )
Optical_Path['P3'] = prop.ASM_Prop( wavelength = lamb,dx = dx, distance = 1*focal_length , N=s[0] , padding = 0, H=None, device = device)

#Once the Optical Path is defined we can combine it into a single network. 
four_f_model=torch.nn.Sequential( Optical_Path ).to(device)


#View the results for both 'batch' images
f,ax_arr = plt.subplots(2,2,figsize=(12,12))


img = 0

im_1 = ax_arr[0,0].imshow(( ( field_in[img,0,:,:]) ).abs().cpu().detach())
ax_arr[0,1].imshow(( four_f_model( field_in)[img,0,:,:] ).abs().cpu().detach())

ax_arr[0,0].set_title('Input field intensity')
ax_arr[0,1].set_title('Output intensity of 4F system with aperture')

ax_arr[0,0].axis('off')
ax_arr[0,1].axis('off')

img = 1

ax_arr[1,0].imshow(( ( field_in[img,0,:,:]) ).abs().cpu().detach())
ax_arr[1,1].imshow(( four_f_model( field_in)[img,0,:,:] ).abs().cpu().detach())

ax_arr[1,0].set_title('Input field intensity')
ax_arr[1,1].set_title('Output intensity of 4F system with aperture')

ax_arr[1,0].axis('off')
ax_arr[1,1].axis('off')


plt.show()