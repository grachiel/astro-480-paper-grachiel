#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: RR-Lyr_Reductions.py
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


def create_median_bias_RR(bias_list, median_bias_filename):
    # - - - - the fits open file process has been removed; as processing zip files (and also for the sake of clarity) the zip opening files has been moved to the full reduction function. Here, focusing on extracting data from the files. NOTE median bias, dark, flat files ARE saved as fits files because they are required for the CCD reduction pipeline
    bias_arrays = [data for data, header in bias_list]

    # - - - - Sigma crippling algorythm to remove outliers, and then combining the arrays as done in lecture - - - - 
    bias_arrays_masked = sigma_clip(bias_arrays, cenfunc='median', sigma=3, axis=0)
    median_bias = np.ma.getdata(np.median(bias_arrays_masked, axis=0)) # np.ma.getdata inserted in order to make the masked array properly save

    # Here is some code to create a new FITS file from the resulting median bias frame.
    # You can replace the header with something more meaningful with information.
    primary = fits.PrimaryHDU(data=median_bias, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(median_bias_filename, overwrite=True)

    return median_bias



def create_median_dark_RR(dark_list, bias_filename, median_dark_filename):

    # - - - - Reading median bias frame - - - - 
    #bias_file = fits.open('median_bias_filename.fit') # comment out for input
    bias_file = fits.open(bias_filename)
    median_bias_read_array = bias_file[0].data
    bias_file.close()

    # - - - - Creating a 2d array - - - -
    # just like in bias, re-adapted for the new zip method. doing the same data manipulations as in create_median_dark
    dark_data_scaled = []
    for data, header in dark_list:        
        # - - - -subtracting the bias frame from each dark image
        dark_no_bias = data - median_bias_read_array 
        # - - - - getting exposure time from the header and subtracting
        exptime = header['EXPTIME']
        dark_current_per_second = dark_no_bias / exptime
        # - - - - creating final 2d arrays
        dark_data_scaled.append(dark_current_per_second)

    # - - - - Sigma crippling algorythm to combine and remove outliers - - - - 
    dark_data_scaled_masked = sigma_clip(dark_data_scaled, cenfunc='median', sigma=3, axis=0)
    # - - - - Creating a median array and making sure the masking is properly taken account of to prevent storage errors
    median_dark = np.ma.getdata(np.median(dark_data_scaled_masked, axis=0).filled(fill_value=0))


    # - - - - Save  - - - - 
    header = dark_list[0][1].copy()
    header['BUNIT'] = 'electrons/s'
    primary = fits.PrimaryHDU(data=median_dark, header=header)
    hdul = fits.HDUList([primary])
    #median_dark_filename = 'median_dark_filename.fit' # comment out for input
    hdul.writeto(median_dark_filename, overwrite=True)

    return median_dark


def create_median_flat_RR(
    flat_list,
    bias_filename,
    median_flat_filename,
    dark_filename=None,
):
    # - - - - Checking to see if all filters match - - - - 
    file_filter = flat_list[0][1]['FILTER']
    match = all(header['FILTER'] == file_filter for data, header in flat_list)
    print(f"All filters match: {match}")

    # - - - - Reading median bias frame - - - - 
    bias_file = fits.open(bias_filename) # comment out for input
    median_bias_read_array = bias_file[0].data
    bias_file.close()

    # - - - - Creating 2d arrays - - - - 
    # just like in bias, re-adapted for the new zip method. doing the same data manipulations as in create_median_flat
    flat_array = []
    for data, header in flat_list:        
        # Subtracting the bias frame from each flat image
        flat_no_bias = data - median_bias_read_array
        # Create actual array
        flat_array.append(flat_no_bias)
            
    # - - - - Sigma crippling algorythm to remove outliers - - - - 
    flat_array_masked = sigma_clip(flat_array, cenfunc='median', sigma=3, axis=0)
    # - - - - Creating a median array and making sure the masking is properly taken account of to prevent storage errors.
    un_normalized_median_flat = np.ma.getdata(np.median(flat_array_masked, axis=0))

    # - - - - Creating normalized flat by dividing the median flat by the median value- - - - 
    median_flat = un_normalized_median_flat / np.median(un_normalized_median_flat)

    # - - - - Saving to fit - - - - 
    header = flat_list[0][1].copy()
    primary = fits.PrimaryHDU(data=median_flat, header=header)
    hdul = fits.HDUList([primary])
    #median_flat_filename = 'median_flat_filename.fit' # comment out for input
    hdul.writeto(median_flat_filename, overwrite=True)

    return median_flat

def reduce_science_frame_RR(
    science_data,
    science_header,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename,
):
    # - - - - Reading the science files that went through the load fits from zip function into an array
    science_read_array = science_data
 

    # - - - - Reading median bias from filename
    with fits.open(median_bias_filename) as bias_file:
        median_bias_read_array = bias_file[0].data
    # - - - - Reading median flat
    with fits.open(median_flat_filename) as flat_file:
        median_flat_read_array = flat_file[0].data
    # - - - - Reading median dark
    with fits.open(median_dark_filename) as dark_file:
        median_dark_read_array = dark_file[0].data
    # - - - - Subtracting bias frame from the science frame
    science_no_bias = science_read_array - median_bias_read_array # for mask s_

    # - - - - Multiplying the dark array by the science frame's exposure time
    exptime = science_header['EXPTIME']
    dark_with_exptime = median_dark_read_array * exptime
    # - - - - Subtracting dark (with exposure time) from science frame
    science_no_bias_no_dark = science_no_bias - dark_with_exptime

    # - - - - Correcting the science frame with the flat: dividing as seen in the course website
    reduced_science = science_no_bias_no_dark / median_flat_read_array
    # removing cosmic rays: optional so skipped

    # - - - - Saving final file as reduced_science_filename # commented out the saving process as there are around 500 RR Lyrae files and I do not want to save every science file
    #header = science_filename[0].header
    #primary = fits.PrimaryHDU(data=reduced_science, header=header)
    #hdul = fits.HDUList([primary])
    #hdul.writeto(reduced_science_filename, overwrite=True)

    return reduced_science

def calculate_gain_RR(files, median_bias_filename): # kept effectively the same; removed comments from calculate_gain for the sake of clarity, but refer back for gain specific comments
    # - - - - Read median bias
    with fits.open(median_bias_filename) as hdul:
        median_bias = hdul[0].data
        
    # - - - - Read and process flats
    flat1_data, flat1_header = files[0]
    flat2_data, flat2_header = files[1]
    
    # - - - - Calculate mean signal
    mean_signal = np.mean((flat1_data + flat2_data)/2)
    
    # - - - - Calculate variance of the difference
    var_diff = np.var(flat1_data - flat2_data)
    
    # - - - - Gain calculation
    gain = mean_signal / (var_diff / 2)
    
    return gain * u.electron/u.adu


def calculate_readout_noise_RR(files, gain): # kept effectively the same; removed comments from calculate_gain for the sake of clarity, but refer back for gain specific comments
    # - - - - Getting the variation of the bias - - - - 
    bias_read_arrays = []
    for data, header in files:
        bias_clean = data - np.median(data, axis=0)
        bias_read_arrays.append(bias_clean)
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
    
def do_aperture_photometry_RR(
    image_data,
    image_header,
    radii,
    sky_radius_in,
    sky_annulus_width,
):
    # - - - - Reading image - - - - 
    #image_file = fits.open(image) # removing the fits open function because the run reduction pipeline will already send it reduced science files
    
    # - - - - Setting up data, and preparing for unit of flux - - - - 
    data = u.Quantity(image_data, unit='adu')
    time_obs = Time(image_header['DATE-OBS'])

    
    # - - - - NEW: Finding the star central to the image, strict enough to filter out other stars without removing original. See do_aperture_photometry for comment - - - - 
    #center_x, center_y = 513, 513
    #cropped_data = data[center_y-100:center_y+100, center_x-100:center_x+100]
    #peaks = find_peaks(data=cropped_data.value, threshold=1000, box_size=11)
    #x_star = peaks['x_peak'][0] + center_x - 100
    #y_star = peaks['y_peak'][0] + center_y - 100    
    #positions = [(x_star, y_star)]
    
    try: # I was encountering an error with my original method of locating the central star. Here, I coded alternatives but kept the first iteration (still used in the do_aperture_photometry function) for convenience
        # - - - - automatic detection method: locating the central star from the mean flux and standard deviation to define the find_peaks function (and threshold of that function)
        mean_val = np.mean(image_data)
        std_val = np.std(image_data)
        threshold = mean_val + 5*std_val  # Dynamic threshold
        peaks = find_peaks(data=image_data, threshold=threshold, box_size=11)
        
        if len(peaks) == 0:  # I was encountering errors running on my initial method from do_aperture_photometry so I attempted the previous method. However, still got some issues, So I attempted the following if else and except. I believe the issue was the cloud coverage at the end of the night, because the central star effectively falls to roughly the same flux as the rest of the sky on all frames at this part of the night.
            center_x, center_y = 513, 513 
            positions = [(center_x, center_y)]
        else:
            brightest = np.argmax(peaks['peak_value']) # Get brightest peak of the whole dataset. In the frames I checked for RR Lyrae no stars were brighter than the target, but this method would fail if that were not the case. I made sure to double-check the frames at the end of the night, but as you will see later, I removed the last 30 frames when the cloud coverage was too extreme so there is no need to worry about this fall back selecting the wrong star
            x_star = peaks['x_peak'][brightest]
            y_star = peaks['y_peak'][brightest]
            positions = [(x_star, y_star)]
            
    except Exception as e:
        print(f"Star finding failed: {str(e)}") # all else fails situation, to make absolutely sure no error messages would appear and we know if it failed. Completes the try, except arguments.
        center_x, center_y = 513, 513
        positions = [(center_x, center_y)]

    
    
    # - - - - Reading positions as tuples; No changes from do_aperture_photometry, so refer to that function for comment.
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

    # - - - - Setting up a table. Creating a star label to distinguish, and later use in the labels for the plots - - - - 
    optimal_flux = np.argmax(final_fluxes)
    
    return Table({
        'time': [time_obs.mjd], 
        'flux': [float(final_fluxes[optimal_flux].value)]
    })




import os
import zipfile
from astropy.io import fits
from io import BytesIO
from astropy.table import vstack

def load_fits_from_zip(zip_file, file_list):
    # - - - - Load FITS data from given filenames in the zip
    data_list = []
    for name in file_list:
        with zip_file.open(name) as f: # zip open command, running functions nearly identical to the fits open ones seen in the ccd reductions assignment
            hdul = fits.open(BytesIO(f.read()))
            data = hdul[0].data
            header = hdul[0].header
            data_list.append((data, header))
            hdul.close()
    return data_list

def process_observations(zip_path, star, flat_filter, output_dir='/home/jovyan/work/'):
    # - - - - creating the reduction pipeline for RR Lyrae, and merging it with plot_light_curve (which you will notice doesn't have an _RR equivalent)
    median_bias_filename = 'median_bias_28.fits'
    median_flat_filename = 'median_flat_28.fits'
    median_dark_filename = 'median_dark_28.fits'

    # - - - - creating bias/dark/flat lists based on zip files (notice use of load_fits_from_zip files) - - - - 
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
    
        bias_files = [f for f in names if 'Bias_' in f and f.endswith('.fits')]
        dark_files = [f for f in names if 'Dark_' in f and f.endswith('.fits')]
        flat_files = [f for f in names if flat_filter in f and f.endswith('.fits')] # filter specific so that we make sure star and flats for the filter star was observed with match
        science_files = [f for f in names if 'RRLyrae' in f and f.endswith('.fits')]
        
        bias_list = load_fits_from_zip(z, bias_files)
        dark_list = load_fits_from_zip(z, dark_files)
        flat_list = load_fits_from_zip(z, flat_files)

    reduced_science_filename = 'reduced_science_filename.fits'
    science_filename=science_files

    # - - - - running functions that create the median files - - - - 
    median_bias = create_median_bias_RR(bias_list, median_bias_filename)
    median_dark = create_median_dark_RR(dark_list, bias_filename=median_bias_filename, median_dark_filename=median_dark_filename)
    median_flat = create_median_flat_RR(flat_list, bias_filename=median_bias_filename, median_flat_filename=median_flat_filename, dark_filename=None,)
    
    
    radii = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0]
    sky_radius_in = 15.0
    sky_annulus_width = 10.0

    # - - - - Doing reduce_science_frame_RR and do_aperture_photometry_RR on each science zip file. Remember that science didn't save the files. the data is sent to do_aperture_photometry_RR, so the saving is unecessary! - - - - 
    photometry_results = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for science in science_files:
            with z.open(science) as f:
                hdul = fits.open(BytesIO(f.read()))
                reduced_science = reduce_science_frame_RR(science_data=hdul[0].data, science_header=hdul[0].header,median_bias_filename=median_bias_filename,median_flat_filename=median_flat_filename,median_dark_filename=median_dark_filename,reduced_science_filename=None)        
                ap_phot_table = do_aperture_photometry_RR(image_data=reduced_science, image_header=hdul[0].header,radii=radii, sky_radius_in=sky_radius_in, sky_annulus_width=sky_annulus_width)
                photometry_results.append(ap_phot_table)
                base_filename = os.path.basename(science)
                hdul.close()
    # - - - - Creating a table out of the results, identical function to when I was making the flux_table. 
    final_table = vstack(photometry_results)
    
    calculate_readout_noise_RR(files=bias_list[:2], gain=calculate_gain_RR(files=flat_list, median_bias_filename=median_bias_filename)) # reporting readout noise for the reduction
    print(calculate_gain_RR(files=flat_list[:2], median_bias_filename=median_bias_filename)) # reporting gain for the reduction

    
    # - - - - Everything below here is identical in function to the plot_light_curve function, just using final table as established above. Check plot_light_curve for commentary - - - -
    final_table.sort('time')
    final_table = final_table[:-30] # removing last 30 frames where cloud coverage is too much and flux drops towards zero
    
    flux = final_table['flux']
    final_table['norm_flux'] = flux / np.median(flux)
    clipped = sigma_clip(final_table['norm_flux'], sigma=3)
    final_table = final_table[~clipped.mask]

    t_start = Time(final_table['time'][0], format='mjd')
    final_table['hours'] = [(Time(t, format='mjd') - t_start).sec / 3600 
                          for t in final_table['time']]
    plt.plot(final_table['hours'], final_table['norm_flux'])
    plt.ylabel('Normalized Flux (ADU)')
    plt.xlabel(f'Time since beginning of observation (hours)')
    plt.title(f'RRLyrae Light Curve')
    plt.tight_layout()
    plt.savefig(f'RR Lyrae Light Curve')
    plt.show()
    return final_table