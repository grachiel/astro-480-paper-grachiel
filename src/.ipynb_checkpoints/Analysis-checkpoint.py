#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: Analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.time import Time
from astropy.io import fits
import astropy.units as u
import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.aperture import ApertureStats
from astropy.table import QTable, Table, Column
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from photutils.detection import find_peaks



def do_aperture_photometry(
    image,
    radii,
    sky_radius_in,
    sky_annulus_width,
):
    # - - - - Reading image - - - - 
    image_file = fits.open(image)
    image_read_array = image_file[0].data
    # - - - - Setting up data, and preparing for unit of flux - - - - 
    data = u.Quantity(image_read_array, unit='adu')
    time_obs = Time(image_file[0].header['DATE-OBS'])

    
    # - - - - NEW independent of CCD assignment: Finding the star central to the image, strict enough to filter out other stars without removing original - - - - 
    center_x, center_y = 513, 513 # 513 determined by zooming into central CCD pixel on ARCSAT output files. Note that this means this is only compatible with our files, as different pixel centers make this line inaccurate
    cropped_data = data[center_y-100:center_y+100, center_x-100:center_x+100] # creating region guarenteed to only contain central star
    peaks = find_peaks(data=cropped_data.value, threshold=200, box_size=11) # using find_peaks... to find peaks (flux). threshold is set at 200 since we are dealing with variable stars. Some files' central star flux gets close to the background with cloud coverage 
    x_star = peaks['x_peak'][0] + center_x - 100 # determining the x and y positions of the star based on the highest flux vale and center of the ccd (-100s to convert back into the region of the entire ccd)
    y_star = peaks['y_peak'][0] + center_y - 100    
    positions = [(x_star, y_star)]
    
    
    # - - - - Reading positions as tuples
    x_y_positions = tuple(map(tuple, positions))
    
    # - - - - defining apertures with each position and radius, using for loop to isolate radius. Using CircularAperture due to dealing with x,y pixel positions - - - - 
    apertures = [CircularAperture(x_y_positions, r=r) for r in radii]
    
    # - - - - defining the sky also with CircularAnnulus, due to using pixel measurments. r_in defined by sky_radius_in, and r_out using sky_radius_in plus the width in order to return the combined radius - - - -  
    sky_annulus = CircularAnnulus(x_y_positions, r_in = sky_radius_in, r_out = sky_radius_in + sky_annulus_width)
    
    # - - - - getting the median value of the data, to be used as the background sky (once scaled to area of the aperture) - - - - 
    sky_stats = ApertureStats(data, sky_annulus)
    sky_values = sky_stats.median

    # - - - - for each aperture defined by each radius, first measuring the flux through aperture_photometry as hinted, then subtracting the sky background found earlier (scaled to area), and finally creating a list of final sky-less fluxes - - - - 
    final_fluxes = []
    for ap in apertures:
        measured_flux = aperture_photometry(data, ap)['aperture_sum']
        flux_without_sky_background = measured_flux - (sky_values * ap.area)
        final_fluxes.append(flux_without_sky_background)

    # - - - - Setting up a table. Previously flux was taken for each aperture. Since my new goal is to plot one point for each star, I only want the maximum flux value (represents the truest value for our data and purpouses) - - - - 
    optimal_flux = np.argmax(final_fluxes)
    
    return Table({
        'time': [time_obs.mjd], # takes time from the header, at the beginning of the function, and the peak flux of the data, creating a 1x2 table specfic to the input science image
        'flux': [float(final_fluxes[optimal_flux].value)] 
    })




from os import listdir
from os.path import isfile, join
from astropy.table import vstack


def plot_light_curve(mypath, start_with): 
    # - - - - creates a simple time vs flux table based on every science file with the same star name (all data used was under one filter per star) - - - - 
    # - - - - defining variables to pass to do_aperture_photometry. Borrowed from the CCD assignment.
    data_list = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith(start_with) and f.lower().endswith('.fits')]
    radii = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0]
    sky_radius_in = 15.0
    sky_annulus_width = 10.0

    plt.figure(figsize=(14, 7))
    
    # - - - - Creating a table by running do_aperture_photometry on every science image in data_list. all 1x2 tables are combined with vstack to make a flux table for the star over the entire observing period - - - -
    flux_table = []
    for data_file in data_list:
            flux_data = do_aperture_photometry(data_file, radii, sky_radius_in, sky_annulus_width)
            flux_table.append(flux_data)
    flux_table = Table(vstack(flux_table))
    flux_table.sort('time') # Making the data chronological, according to the time listed in the table
    t_start = Time(flux_table['time'][0], format='mjd') # getting the time of the first entry. 
    flux_table['hours'] = [(Time(t, format='mjd') - t_start).sec / 3600 
                          for t in flux_table['time']] # Personal choice to make all tables start at 0, so subtracting the start time from each entry. Also converting to hours for interpretation.
    flux = flux_table['flux']
    flux_table['norm_flux'] = flux / np.median(flux) # normalizing the flux for interpretation
    plt.plot(flux_table['hours'], flux_table['norm_flux'], 'o-')
    plt.ylabel('Normalized Flux (ADU)')
    plt.xlabel(f'Time since beginning of observation (hours)')
    plt.title(f'{(mypath)[18:-1]} Light Curve') # making the title isolate the star name based on the path used. NOTE this will only work as intended for my specific attempt, since I ran everything out of my own jupytr with my own name and directories
    plt.tight_layout()
    plt.savefig(f'{(mypath)[18:-1]} Light Curve') # same as plot title
    plt.show()
    return flux_table # output shows the plot and returns the flux table, so later I can run this in a different function and use the table, output that function's table, and output this function's table all at once. 