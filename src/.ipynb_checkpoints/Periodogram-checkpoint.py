#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: Periodogram.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable, Table, Column
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table



def get_periodogram(t, y_obs, name):
    # - - - - - - - - Part 2: Lomb_Scargle to get the period - - - - - - - - 
    # based on example used in ASTR 324: just using standard equations and plugging them into standard functions
    from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
    from astropy.timeseries import LombScargle
    
    # - - - - for Lomb Scargle, setting up desired statistical variables - - - -
    
    sigma_y = np.std(y_obs) # standard deviation of the flux. Idea is to take the flux column of the flux vs time table I created (so y_obs will be the flux component and t the time component)
    max_time = (t[-1])/24 # to be used for a dynamic x-axis bound and setting up the range for the period linspace (and hence why max_time is divided by 24, to remain consistent with the scale of the linspace (0 to 1, 24h ->1)
    period = np.linspace(0.03, max_time, 10000) # setting up linspace for the period
    omega = 2 * np.pi / period 

    # - - - - Setting up the Lomb Scargle function. This version is based on the astropy version, while the original code I made was based on the astroML version (astroML version was not working in this notebook so the alternative astropy version was created). Only commenting out the astroML version for preference
    #PS = lomb_scargle(t, y_obs, omega, generalized=True) #astroML version
    ls = LombScargle(t, y_obs, dy=sigma_y) # astropy version
    PS=ls.power(omega) # asrtropy Version
    P_fit = period[np.argmax(PS)] # Lomb Scargle predicted period based on the maximum peak in the data
    #D = lomb_scargle_bootstrap(t, y_obs, omega, generalized=True,N_bootstraps=500, random_state=0) # astroML version
    n_bootstraps=500 # part of astroML version
    D = np.zeros((n_bootstraps, len(omega))) # the following lines are all part of the astropy version. Setting up lomb_scargle_bootstrap function based on bootstraps manually becaue the function does not traditionally accept N_bootstraps as a variable
    for i in range(n_bootstraps): # "..."
        np.random.seed(i) # "..."
        y_shuffle = np.random.permutation(y_obs) # "..."
        D[i] = LombScargle(t, y_shuffle, dy=sigma_y).power(omega) # "..."
    sig1, sig5 = np.percentile(D, [99, 95]) # getting the probabilistic regions, to later be used to visually demonstrate where the percentiles lie
    
    plt.figure(figsize=(14, 7))
    plt.plot(period, PS, '-', c='black', lw=1, zorder=1)
    plt.plot([period[0], period[-1]], [sig1, sig1], ':', c='black', label="99% significance level") # making 99 significance level line based on the data
    plt.plot([period[0], period[-1]], [sig5, sig5], '-.', c='black', label="95% significance level") # same but for 95
    plt.xlabel('Period (days)')
    plt.ylabel('Lomb-Scargle Power')
    plt.legend()
    caption = (f"The most likely period (the highest periodogram peak), P_fit, is {P_fit:.3f} days")
    plt.text(0.5, 0.95, caption, transform=plt.gca().transAxes, ha='center', fontsize=12)
    plt.savefig(f'{name} periodogram.png')
    plt.show()
