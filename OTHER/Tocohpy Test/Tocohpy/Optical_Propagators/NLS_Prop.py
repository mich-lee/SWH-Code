import torch
import numpy as np
import math    
import warnings



##########
#### This Package contains functions to propagate optical fields t 
#### Propagation method for propagating though a nonlinear crystal using a Fourier Split step method 
#### and solving the nonlinear shrodinger method 
####Author: Lionel Fiske 
####Last update 9/15/2021
#####

import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 



class NLS_Prop(torch.nn.Module):
    
    def __init__(self, wavelength, dx, distance , N , H=None, padding = 1/2 , background_index = 1, dtype = torch.complex64 ,device = torch.device("cpu")):
        """
        Angular Spectrum method which is the exact solution to the Linear portion of the NLS equation
        Derived by hand. 
        
        Parameters
        ==========
                           
        wavelength        : float
                           Wavelength of light
        
        dx                : float
                           Pixel Size (same units as wavelength)
                           
        distance          : float
                          Propagation Distance
       
        N                  : int 
                          Simulation size
        
                                     
        H                 : torch.complex128
                          Simulation size
                         
        padding           : float
                          percent of domain to pad for simulation
                                           
                         
                                 
        device             :  torch device
                           
        """
        
        
        super(NLS_Prop, self).__init__()
        
        self.wavelength = wavelength
        self.dx = dx
        self.distance = distance
        self.device = device
        self.padding = padding
        self.background_index = background_index
        
    #Compute H if not precomputed
        if H == None:
        #Max Frequency
            k =  2*self.background_index*np.pi/self.wavelength

            # Get the shape for processing
            Nx =int( (1 + 2*padding) * N)
            Ny =int( (1 + 2*padding) * N)

            kx_max =  .5/np.abs(self.dx)
            ky_max = kx_max

            # Compute 1D-grid for both directions
            kx = 2*np.pi*torch.linspace(-kx_max,kx_max,Nx, device = self.device);
            ky = 2*np.pi*torch.linspace(-ky_max,ky_max,Ny, device = self.device);

            # Compute the meshgrid to represent Kx Kv coordinates
            Kx,Ky = torch.meshgrid(kx, ky)

            # Compute the ASM kernel
            ang = ( (self.distance / k /2 ) *  ( Kx**2 + Ky**2 ) ).type(dtype)
            H = torch.cos(ang) + 1j*torch.sin(ang) 


    
        self.H = torch.unsqueeze(torch.unsqueeze(  H , dim=0 ) ,dim=0) .to(self.device) 

        
    def forward(self, field):
        """
         Applies propagation kernel

        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).
                       
        """
        

        #Grab size
        Nx =  field.shape[-2]
        Ny =  field.shape[-1]

        # Save for later crop
        Nx_old = int(Nx)
        Ny_old = int(Ny)

        # Pad the image for avoiding convolution artifacts
        Ein = torch.nn.functional.pad(field, (int( self.padding * Nx_old),int(self.padding *Nx_old),int(self.padding *Ny_old),int(self.padding*Ny_old)), mode='constant', value=0)

        
        # Apply the ASM kernel
        Ein_fft = ft2( Ein )  
        Eout_no_pad =ift2(Ein_fft*self.H) 
        
        
        # Unpad the image the original size
        Eout = torch.nn.functional.pad(Eout_no_pad, (-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Nx_old)), mode='constant', value=0)

        if field.ndim == 2:
            return Eout.squeeze()
        else:
            return Eout 

        return Eout 

####
