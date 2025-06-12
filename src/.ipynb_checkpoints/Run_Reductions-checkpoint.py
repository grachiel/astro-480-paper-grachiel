#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: Run_Reductions.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
from os import listdir
from os.path import isfile, join
from os.path import splitext
from astropy.io import fits
from astropy.stats import sigma_clip
import astropy.units as u
import numpy as np
from astropy.io.fits import getheader
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval


def create_median_bias(bias_list, median_bias_filename):

    # - - - - Making the list 2D arrays, opening the files - - - - 
    bias_arrays = []
    for bias_file in bias_list:
        with fits.open(bias_file) as hdul:
            bias_data = hdul[0].data
            # - - - - cropping the frame - - - - 
            s_bias_data = bias_data #[2048:3072, 2048:3072]
            bias_arrays.append(s_bias_data) #for mask, s_
            # - - - - Making a header with a brief description of what was done - - - - 
            header = hdul[0].header.copy()
            header['HISTORY'] = 'Median bias created with same units'
        
    # - - - - Sigma crippling algorythm to remove outliers, and then combining the arrays as done in lecture - - - - 
    bias_arrays_masked = sigma_clip(bias_arrays, cenfunc='median', sigma=3, axis=0)
    median_bias = np.ma.getdata(np.median(bias_arrays_masked, axis=0)) # np.ma.getdata inserted in order to make the masked array properly save

    # Here is some code to create a new FITS file from the resulting median bias frame.
    # You can replace the header with something more meaningful with information.
    primary = fits.PrimaryHDU(data=median_bias, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(median_bias_filename, overwrite=True)

    return median_bias

def create_median_dark(dark_list, bias_filename, median_dark_filename):

    # - - - - Reading median bias frame - - - - 
    #bias_file = fits.open('median_bias_filename.fit') # comment out for input
    bias_file = fits.open(bias_filename)
    median_bias_read_array = bias_file[0].data

    # - - - - Creating a 2d array - - - - 
    dark_data_scaled = []
    for dark_file in dark_list:
        with fits.open(dark_file) as hdul:
            dark_data = hdul[0].data
            s_dark_data = dark_data #[2048:3072, 2048:3072]
            # - - - -subtracting the bias frame from each dark image
            dark_no_bias = s_dark_data - median_bias_read_array # for mask, s_
            # - - - - getting exposure time from the header and subtracting
            exptime = hdul[0].header['EXPTIME']
            dark_current_per_second = dark_no_bias / exptime
            # - - - - creating final 2d arrays
            dark_data_scaled.append(dark_current_per_second)
            header = hdul[0].header.copy() # adjust later, want to update header
            header['BUNIT'] = 'electrons/s'

    # - - - - Sigma crippling algorythm to combine and remove outliers - - - - 
    dark_data_scaled_masked = sigma_clip(dark_data_scaled, cenfunc='median', sigma=3, axis=0)
    # - - - - Creating a median array and making sure the masking is properly taken account of to prevent storage errors
    median_dark = np.ma.getdata(np.median(dark_data_scaled_masked, axis=0).filled(fill_value=0))


    # - - - - Save to FITS - - - - 
    primary = fits.PrimaryHDU(data=median_dark, header=header)
    hdul = fits.HDUList([primary])
    #median_dark_filename = 'median_dark_filename.fit' # comment out for input
    hdul.writeto(median_dark_filename, overwrite=True)

    return median_dark


def create_median_flat(
    flat_list,
    bias_filename,
    median_flat_filename,
    dark_filename=None,
):
    # - - - - Checking to see if all filters match - - - - 
    file_filter = fits.getheader(flat_list[0])['FILTER']
    match = all(fits.getheader(f)['FILTER'] == file_filter for f in flat_list)
    print(f"All filters match: {match}")

    # - - - - Reading median bias frame - - - - 
    bias_file = fits.open(bias_filename) # comment out for input
    median_bias_read_array = bias_file[0].data

    # - - - - Creating 2d arrays - - - - 
    flat_array = []
    for flat_file in flat_list:
        with fits.open(flat_file) as hdul:
            flat_data = hdul[0].data
            s_flat_data = flat_data #[2048:3072, 2048:3072]
            # - - - - subtracting the bias frame from each dark image
            flat_no_bias = s_flat_data - median_bias_read_array #for mask, s_
            # - - - - Create actual array
            flat_array.append(flat_no_bias)
            header = hdul[0].header.copy() # edit later for specificity
            
    # - - - - Sigma crippling algorythm to remove outliers - - - - 
    flat_array_masked = sigma_clip(flat_array, cenfunc='median', sigma=3, axis=0)
    # - - - - Creating a median array and making sure the masking is properly taken account of to prevent storage errors.
    un_normalized_median_flat = np.ma.getdata(np.median(flat_array_masked, axis=0).filled(fill_value=0))

    # - - - - Creating normalized flat by dividing the median flat by the median value- - - - 
    median_flat = un_normalized_median_flat / np.median(un_normalized_median_flat)

    # - - - - Saving to fit - - - - 
    primary = fits.PrimaryHDU(data=median_flat, header=header)
    hdul = fits.HDUList([primary])
    #median_flat_filename = 'median_flat_filename.fit' # comment out for input
    hdul.writeto(median_flat_filename, overwrite=True)

    return median_flat

def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename="reduced_science.fits",
):
    # - - - - Reading a science file from name
    science_file = fits.open(science_filename)
    science_read_array = science_file[0].data

    # - - - - Reading median bias from filename
    bias_file = fits.open(median_bias_filename)
    median_bias_read_array = bias_file[0].data

    # - - - - Reading median flat
    flat_file = fits.open(median_flat_filename)
    median_flat_read_array = flat_file[0].data

    # - - - - Reading median dark
    dark_file = fits.open(median_dark_filename)
    median_dark_read_array = dark_file[0].data

    # - - - - Subtracting bias frame from the science frame
    s_science_read_array = science_read_array #[2048:3072, 2048:3072] # Cropping Science array to 1024, centered on image (same for bias, dark, flat)
    science_no_bias = s_science_read_array - median_bias_read_array # for mask s_

    # - - - - Multiplying the dark array by the science frame's exposure time
    exptime = science_file[0].header['EXPTIME']
    dark_with_exptime = median_dark_read_array * exptime
    # - - - - Subtracting dark (with exposure time) from science frame
    science_no_bias_no_dark = science_no_bias - dark_with_exptime

    # - - - - Correcting the science frame with the flat: dividing as seen in the course website
    normalized_flat = median_flat_read_array / np.median(median_flat_read_array)
    reduced_science = science_no_bias_no_dark / normalized_flat

    # removing cosmic rays: optional so skipped

    # - - - - Saving final file as reduced_science_filename
    header = science_file[0].header
    primary = fits.PrimaryHDU(data=reduced_science, header=header)
    hdul = fits.HDUList([primary])
    hdul.writeto(reduced_science_filename, overwrite=True)

    return reduced_science


def calculate_gain(files, median_bias_filename):
    # - - - - Read median bias
    with fits.open(median_bias_filename) as hdul:
        median_bias = hdul[0].data
    
    # - - - - Read and process flats. Subtract bias from the flats
    with fits.open(files[0]) as hdul1, fits.open(files[1]) as hdul2:
        flat1 = hdul1[0].data - median_bias
        flat2 = hdul2[0].data - median_bias
 
    # - - - - Calculate mean signal by taking the mean of the average of the flats (should return one value)
    mean_signal = np.mean((flat1 + flat2)/2)
    
    # - - - - Calculate variance of the difference (to get counts per pixel)
    var_diff = np.var(flat1 - flat2)
    
    # - - - - Gain calculation
    gain = mean_signal / (var_diff / 2)  # in e-/ADU; see return statement.
    
    return gain * u.electron/u.adu


def calculate_readout_noise(files, gain):
    # - - - - Getting the variation of the bias - - - - 
    bias_read_arrays = []
    for f in files:
        with fits.open(f) as hdul:
            bias_data = hdul[0].data
            s_bias_data = bias_data#[2048:3072, 2048:3072]
            bias_clean = s_bias_data - np.median(s_bias_data, axis=0) # Was getting unreasonable error, so subtracting the median bias arrays in an attempt to remove the noise that is making what should be a mathematically accurate function incorrect 

            bias_read_arrays.append(bias_clean)

    # - - - - setting bias_std_diff, using the denominator code from the gain function but redefining it for bias. Paying special care to be std, not var()/2 - - - -         
    if len(bias_read_arrays) > 2:
        bias_difference = bias_read_arrays[0]-np.mean(bias_read_arrays[1:], axis=0)
    else:
        bias_difference = bias_read_arrays[0]-bias_read_arrays[-1]
    bias_std_diff = (np.std(bias_difference)) * u.adu

    # - - - - Readnoise equation - - - 
    readout_noise = gain * bias_std_diff / np.sqrt(2)

    print(readout_noise)
    
def run_reduction(data_dir, star, flat_filter):
    # - - - - following the same reduction process as in the ccd reductions assignment. FIRST, reading the file and creating bias, dark, flat (for target band), and science (for target star) lists 
    bias_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.startswith('Bias_') and f.lower().endswith('.fits')]
    dark_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.startswith('Dark_') and f.lower().endswith('.fits')]
    flat_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.startswith(flat_filter) and f.lower().endswith('.fits')]
    science_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.startswith(star) and f.lower().endswith('.fits')]
    # - - - - setting up filenames just like in the assignment
    median_bias_filename = 'median_bias_filename.fits'
    bias_filename = 'median_bias_filename.fits'
    median_flat_filename = 'median_flat_filename.fits'
    median_dark_filename = 'median_dark_filename.fits'
    reduced_science_filename = 'reduced_science_filename.fits'
    science_filename=science_list

    # - - - - Running reduction process like in the assignment
    create_median_bias(bias_list, median_bias_filename)
    create_median_dark(dark_list, bias_filename=median_bias_filename, median_dark_filename=median_dark_filename)
    create_median_flat(flat_list, bias_filename=median_bias_filename, median_flat_filename=median_flat_filename, dark_filename=None,)
    # - - - - For science files, running it such that it processes all science files instead of one (as was the case in the ccd assignment)
    for science_file in science_list:
        original_name = os.path.basename(science_file)
        output_name = f'{original_name}_out.fits' # creating an output name to save the reduced files to, in order to perform analysis on. When I rewrite everything for RR Lyrae in zip files, I do not save the files
        reduce_science_frame(science_file, median_bias_filename, median_flat_filename, median_dark_filename,reduced_science_filename=output_name)
    calculate_readout_noise(files=bias_list, gain=calculate_gain(files=flat_list, median_bias_filename='median_bias_filename.fits')) # function adjusted to only use two bias files, so no need in defining a set of two files here.
    print(calculate_gain(files=flat_list, median_bias_filename='median_bias_filename.fits')) # calculate_readout_noise already prints itself, so print statement for gain moved here. Gain function adjusted to only read two flat files, so no need in defining a set of two files here. 
    
    

    
