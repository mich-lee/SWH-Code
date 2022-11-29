import torch
import numpy as np
import math    
import warnings


##########
#### This Package contains functions to propagate optical fields 
#### An angular spectrum method which allows for a new gridspacing at the output plane 
#### Author: Florian Schiffers  
#### Last update 09/15/2021 by Lionel Fiske - checked comments 
#####

import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 


class ASM_Prop_NC(torch.nn.Module):
    
    def __init__(self, N, wavelength, dx, dx_new, distance, dtype=torch.complex128, device = torch.device("cpu")):
        """
        From Numerical Simulation of Optical Systems by Jason Schmidt
        Listing 6.5 page 102
        
        Example of evaluating the Fresnel diffraction integral using the angular-spectrum method
        
        NOTE: Changes coordinate system and does not pad
        
      Parameters
        ==========
                           
        wavelength        : float
                           Wavelength of light
        
        dx                : float
                           Pixel Size (same units as wavelength)
                           
                
        dx_new             : float
                           Desired output pixel size (same units as wavelength)                   
                           
        distance          : float
                          Propagation Distance
        
        N                  : int 
                          Simulation size
                                     
          
        device            :torch device
                           
        """
        super(ASM_Prop_NC, self).__init__()
        
        self.dx = dx #input plane spacing
        self.dx_new = dx_new #output plane spacing
        # optical wavevector
        k = 2*np.pi/wavelength

        #source-plane coordinates
        x = torch.linspace(-N/2, N/2 - 1,N) * self.dx
        y = torch.linspace(-N/2, N/2 - 1,N) * self.dx

        Y1, X1 = torch.meshgrid(y,x)

        R2 = X1**2 + Y1**2;
        
        # spatial frequencies (of source plane)
        self.dkx = 1 / (N*self.dx)
        self.dky = 1 / (N*self.dx)

        kx = torch.linspace(-N/2 , N/2 - 1, N) * self.dkx
        ky = torch.linspace(-N/2 , N/2 - 1, N) * self.dky

        Ky, Kx = torch.meshgrid(ky,kx)

        KR2 = Kx**2 + Ky ** 2

        # scaling parameter
        self.m = self.dx_new/self.dx

        # observation-plane coordinates
        x = torch.linspace(-N/2, N/2 - 1,N) * self.dx_new
        y = torch.linspace(-N/2, N/2 - 1,N) * self.dx_new

        Y2, X2 = torch.meshgrid(y,x)

        R2_new = X2**2 + Y2**2;
        
        # quadratic phase factors
        self.Q1 = torch.exp(1j*k/2*(1-self.m)/distance*R2).to(device).type(dtype)
        self.Q2 = torch.exp(-1j*np.pi**2*2*distance/self.m/k*KR2).to(device).type(dtype)
        self.Q3 = torch.exp(1j*k/2*(self.m-1)/(self.m*distance)*R2_new).to(device).type(dtype)

        
    def forward(self, field):
        """
        Apply angular spectrum
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).
                       
        """
        
        out =  self.Q3 * ift2(self.Q2 * ft2(self.Q1 * field / self.m, self.dx), self.dkx)
        
        if field.ndim == 2:
            return out.squeeze()
        else:
            return out 


