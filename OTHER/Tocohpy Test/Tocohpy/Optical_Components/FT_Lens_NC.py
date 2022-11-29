##########
#### This Package contains components to build optical setups. 
#### Lens in Fourier Transform configuration with new coordinates
####Author: Lionel Fiske 
####Last update 9/15/2021
#####


import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 




class FT_Lens_NC(torch.nn.Module):
    
    def __init__(self, f, wavelength, dx, N, d = None, dtype = torch.complex128 ,  device = torch.device("cpu")  ):
        """
        This takes an image in the rear focal plane and computes the (properly sampled) image in the  Fourier plane
        This implementation rescales the coordinates in the fourier plane and stores the new coordinates as self.image_coords_x and image_coords_y
        adapted from: Numerical simulation of optical wave simulation , Jason Schmidt 
        ==========
        f              : float
                      Focal length of lens (matching units with wavelength)
                           
        wavelength     :   float
                          wavelength


        """
        
        
        super(FT_Lens_NC, self).__init__()

        
        #d is object position behind lens> If none assume object is one focal length 
        if d == None:
            self.d = f
        else:
            self.d = d
            
        #Set internal variables    
        self.device = device    
        self.wavelength = wavelength  
        self.dx =dx 
        self.dx_out =  wavelength * f / (N * dx)
        self.f = f
        self.N = N
        self.kx = torch.linspace(-N/2, N/2, N) / (N*dx)
        self.ky = torch.linspace(-N/2, N/2, N) / (N*dx)
        self.k = 2*np.pi / wavelength
        self.dtype = dtype
        
        # observation plane coordinates
        V,U = torch.meshgrid(wavelength * f * self.kx, wavelength * f * self.ky) 

        self.dx_new = wavelength * f / (N * dx)
        self.prefactor =  (self.dx**2  / (1j*self.wavelength*self.f)  * torch.exp( (1 - self.d/self.f)*1j*self.k/(2*self.f) *(V**2 + U**2))).to(device).type(dtype)
    
    
    def forward(self, field):
        """
        In this function we apply a phase delay which depends on the local intensity of the electric field at that point
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).
                  
        """
        
        out = self.prefactor * ft2(field, norm = 'backward')


        if field.ndim == 2:
            return out.squeeze()
        else:
            return out 

