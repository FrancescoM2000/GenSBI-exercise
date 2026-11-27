import numpy as np
import torch
import matplotlib.pyplot as plt

# Useful functions

def power_law_spectrum(E_edges, gamma):

    if gamma == 1.:
        spectrum = torch.log(E_edges[1:] / E_edges[:-1])
    else:
        spectrum = (
            E_edges[1:]**(1. - gamma)
            - E_edges[:-1]**(1. - gamma)
        ) / (1. - gamma)

    return spectrum / spectrum.sum()

def generate_dummy_maps( n_side , Emin , Emax , n_energy , theta ):
    
    # energy bins
    E_edges = torch.logspace(np.log10(Emin), np.log10(Emax), n_energy + 1)  # GeV
    E_mid = torch.sqrt(E_edges[:-1]*E_edges[1:])

    # Generate dummy but realistic sky maps using torch

    # create 2D grid for pixels' (tokens') coordinates
    # Pixel boundaries
    edges = torch.linspace(-1., 1., n_side + 1)

    # Pixel centers
    u = (edges[:-1] + edges[1:]) / 2
    v = u.clone()

    # Spatial grid
    U, V = torch.meshgrid(u, v, indexing="ij")

    # Coordinate features
    grid = torch.stack([U, V], dim=-1)


    # create energy features
    # first the exposure map: \[f_{ij}=1+\epsilon\cos(\pi U_{ij})\cos(\pi V_{ij}) * g(E)=\left(\frac{E}{E_0}\right)^\alpha. 

    Exposure0 = 3.e11   # cm² s
    epsilon = 0.15
    alpha = 0.1
    E0 = 1.0           # GeV

    exposure_spatial = ( 1. + epsilon * torch.cos(torch.pi * U) * torch.cos(torch.pi * V) )
    exposure_energy = ( E_mid / E0 )**alpha
    exposure = Exposure0 *  torch.unsqueeze(exposure_spatial,dim=-1) * exposure_energy

    # torch.unsqueeze(exposure_spatial,dim=-1)     16 16  1
    # exposure_energy                               1  1  8
    # exposure                                     16 16  8

    # then the galactic foreground map
    sigma_gal = 0.3
    gamma_gal = 2.5
    G0 = 1.e-6  # ph / (cm² s sr)
    foreground_spatial = torch.exp( - V**2 / ( 2. * sigma_gal**2 ))
    foreground_energy = power_law_spectrum(E_edges, gamma_gal)
    foreground = G0 * torch.unsqueeze(foreground_spatial , dim =-1) * foreground_energy

    # Isotropic background
    gamma_iso = 2.3
    I0 = 3.e-7  # ph / (cm² s sr)
    isotropic_spatial = torch.ones(n_side,n_side)
    isotropic_energy = power_law_spectrum(E_edges,gamma_iso)
    isotropic = I0 * torch.unsqueeze( isotropic_spatial , dim=-1) * isotropic_energy

    # Source generation: positioning, photon flux between 1-10 GeV of the source, spectral index of the source
    Nsrc = 5
    Smin = 1.e-10  # ph / (cm² s)
    Smax = 1.e-9   # ph / (cm² s)
    gamma_src = 2.1

    sources_spatial = torch.rand(Nsrc,2)*2. -1.
    sources_flux = Smin + torch.rand(Nsrc) * ( Smax - Smin )
    sources_energy = power_law_spectrum(E_edges , gamma_src)

    sources_flux_energy = torch.unsqueeze(sources_flux , dim=-1) * sources_energy

    # Add a Gaussian 2D PSF, with discrete normalization
    sigma0 = 0.18
    beta = 0.6
    E0 = 1.0  # GeV
    sigma_PSF = sigma0 * ( E_mid / E0 )**(-beta)
    r_sq = ( U.unsqueeze(-1) - sources_spatial[:,0])**2 + ( V.unsqueeze(-1) - sources_spatial[:,1])**2
    PSF = ( 1. / ( 2. * torch.pi * sigma_PSF**2) ) * torch.exp( - r_sq.unsqueeze(-1) / (  2. * sigma_PSF**2 ) )
    PSF_norm = PSF / PSF.sum(dim=(0,1),keepdim=True)

    # Obtain the flux coming from each source in each energy bin
    sources_flux_map = PSF_norm * sources_flux_energy
    sources_map = sources_flux_map.sum(dim=2)

    # simulating photon counts \[
    # theta = torch.tensor([1.2, 0.8, 1.5])
    pixel_solid_angle = ( (10.*torch.pi) / (n_side*180.))**2
    expected_counts = exposure * ( pixel_solid_angle * ( theta[0]*foreground + theta[1]*isotropic ) + theta[2]*sources_map )
    photon_counts = torch.poisson(expected_counts)

    # Checking some informations
    expected_gal = exposure * pixel_solid_angle * theta[0] * foreground
    expected_iso = exposure * pixel_solid_angle * theta[1] * isotropic
    expected_src = exposure * theta[2] * sources_map

    return grid, exposure, foreground, photon_counts
