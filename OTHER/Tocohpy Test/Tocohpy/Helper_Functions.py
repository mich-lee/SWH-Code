##########
#### This Package contains various quality of life functions 
####
####Authors:Florian Schiffers, Lionel Fiske 
####Last update 5/14/2021
#####

from __future__ import print_function
import time
import torch
import torchvision
import numpy as np
import os, time
from PIL import Image
import sys
import math    
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 




def ft2(g, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    
    
    # Save Size for later crop
    Nx_old = int(g.shape[-2])
    Ny_old = int(g.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        g = torch.nn.functional.pad(g, (pad_nx,pad_nx,pad_ny,pad_ny), mode='constant', value=0)
        
    # Compute the Fourier Transform
    out = (delta**2)*torch.fft.fftshift(  torch.fft.fft2(  torch.fft.fftshift(g, dim=(-2,-1))  , dim=(-2,-1), norm=norm)  , dim=(-2,-1))

    if pad == True:
        out = torch.nn.functional.pad(out, (-pad_nx,-pad_nx,-pad_ny,-pad_ny), mode='constant', value=0)

    return out
        
    

def ift2(G, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    
        
    # Save Size for later crop
    Nx_old = int(G.shape[-2])
    Ny_old = int(G.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        G = torch.nn.functional.pad(G, (pad_nx,pad_nx,pad_ny,pad_ny), mode='constant', value=0)
        
    # Compute the Fourier Transform
    out = (delta)**2 * torch.fft.ifftshift(  torch.fft.ifft2(  torch.fft.ifftshift(G, dim=(-2,-1))  , dim=(-2,-1) , norm=norm)  , dim=(-2,-1))

    if pad == True:
        out = torch.nn.functional.pad(out, (-pad_nx,-pad_nx,-pad_ny,-pad_ny), mode='constant', value=0)

    return out
        


# Better colorbar for subplots
# https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

