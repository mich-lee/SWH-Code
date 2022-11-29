import torch
import numpy as np
import math    
import warnings



##########
#### This Paackage contains functions to propagate optical fields 
#### One Step Fresnel transform with new coordinates
####Author: Lionel Fiske , Florian Schiffers
####Last update 9/15/2021 - Lionel Fiske
#####

import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 


class One_Step_Prop_NC(torch.nn.Module):
    
    def __init__(self, wavelength, dx, distance , N , H=None, padding=1/2, dtype=torch.complex128 ,device = torch.device("cpu")):
        """
        One step Fresnel Propagator from Numerical Simulation of Optical Systems by Jason Schmidt Chapter 6 page 91. 
        
        Note: The output of this rescales dx and stores the new grid spacinf in dx_new.
        
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
        super(One_Step_Prop_NC, self).__init__()
        
        #Params 
        self.wavelength = wavelength
        self.dx = dx
        self.dx_new = wavelength*distance/(2*N*dx) #factor of 2 due to padding 
        self.padding = padding
        self.distance = distance
        self.device = device
        self.N = N
        
        
        
        #Compute prop kernel
        if H==None: 
            k = 2*np.pi/wavelength
            print('inner',int( (1+2*self.padding)* N/2)  , N , self.padding)
            #old coords
            spatial_coordinates = torch.linspace(-int( (1+2*self.padding)*N/2)  , int( (1+2*self.padding)*N/2)  , int( (1+2*self.padding)*N)  ) *self.dx 
            
            #new coords
            spatial_coordinates_2 = torch.linspace(-int( (1+2*self.padding)*N/2)   ,int( (1+2*self.padding)*N/2)  ,int( (1+2*self.padding)*N)  ) *self.dx_new

            X,Y  = torch.meshgrid(spatial_coordinates,spatial_coordinates)
            X2,Y2  = torch.meshgrid(spatial_coordinates_2,spatial_coordinates_2)
            
            #Multiplied in 
            self.prefactor =(torch.exp(torch.tensor( 1j* ( k/(2*self.distance) *(X2**2 + Y2**2) ) ) ) * (  self.dx**2 /(1j*self.distance*wavelength) )  ) .to(device).type(dtype)
            #convolved in
            self.H =  1 * torch.exp(1j* k /(2 * self.distance) * (X**2 + Y**2) ).to(device).type(dtype)
                        
        
        
        
        
    def forward(self, field):
        """
        One step Fresnel Propagator from Numerical Simulation of Optical Systems by Jason Schmidt Chapter 6 page 91. 
        
        Note: The output of this rescales dx and stores the new grid spacinf in dx_new.

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
        Ein = torch.nn.functional.pad(field, (int(self.padding * Nx_old),int( self.padding *Nx_old),int(self.padding *Ny_old),int(self.padding *Ny_old)), mode='constant', value=0)

        
        # Apply the ASM kernel
        Ein_fft = self.prefactor*ft2( Ein*self.H , norm = 'backward')

            # Unpad the image the original size
        Eout = torch.nn.functional.pad(Ein_fft, (-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Nx_old)), mode='constant', value=0)


        if field.ndim == 2:
            return Eout.squeeze()
        else:
            return Eout 

