"""
Mark Lundine
This is an experimental script to investigate how much information can be pulled
from satellite shoreline timeseries.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import random
import pandas as pd

def time_array_to_years(datetimes):
    """
    Converts array of timestamps to years
    inputs:
    datetimes (array): array of datetimes
    """
    initial_time = datetimes[0]
    datetimes_seconds = [None]*len(datetimes)
    for i in range(len(datetimes)):
        t = datetimes[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    return datetimes_years

def linear_trend(t, trend_val):
    """
    Applies linear trend to trace
    y = mx+b
    """
    y = trend_val*t
    return y

def sine_pattern(t, A, period_years):
    """
    Applies seasonal pattern of random magnitude to trace
    y = A*sin((2pi/L)*x)
    """
    y = A*np.sin(((2*np.pi*t)/period_years))
    return y

def enso_pattern(t, amplitudes, periods):
    """
    Enso-esque pattern mean of a bunch of sine waves with periods between 3 and 7 years
    """
    y = [None]*len(amplitudes)
    for i in range(len(y)):
        y[i] = amplitudes[i]*np.sin(2*np.pi*t/periods[i])
    y = np.array(y)
    return np.mean(y)

def noise(y, noise_val):
    """
    Applies random noise to trace
    """
    ##Let's make the noise between -20 and 20 m to sort of simulate the uncertainty of satellite shorelines
    noise = np.random.normal(-1*noise_val,noise_val,1)
    y = y+noise
    return y

def apply_NANs(y,
               nan_idxes):
    """
    Randomly throws NANs into a shoreline trace
    """
    y[nan_idxes] = np.nan
    return y
    
def make_matrix(dt):
    """
    Just hardcoding a start and end datetime with a timestep of dt days
    Make a matrix to hold cross-shore position values
    """
    datetimes = np.arange(datetime.datetime(1984,1,1),
                          datetime.datetime(2024,1,1),
                          datetime.timedelta(days=dt)
                          ).astype(datetime.datetime)
    num_transects = len(datetimes)
    ##matrix dimensions
    ##width is longshore distance
    ##height is time
    shoreline_matrix = np.zeros((len(datetimes), num_transects))
    return shoreline_matrix, datetimes

##Initialize stuff
##random.seed(0) #uncomment if you want to keep the randomness the same and play with hard-coded values
dt = 12 #revisit time in days
matrix, datetimes = make_matrix(dt)
num_timesteps = matrix.shape[0]
num_transects = matrix.shape[1]
t = time_array_to_years(datetimes)
x = np.array(range(num_transects))

##setting the random noise amount, here it is +/- 20m
noise_val = 10

##getting a random linear trend between -25 m/year and 25 m/year
trend_val = -1#random.uniform(-25, 25)

##getting a random amplitude for the 6 month cycle between 0 and 20 m
six_month_amplitude = random.uniform(0,20)

##getting a random amplitude for the yearly cycle between 0 and 20 m
yearly_amplitude = random.uniform(0,20)

##getting a random amplitude for the decadal cycle between 0 and 20 m
decadal_amplitude = random.uniform(0,20)

##making ENSO-esque amplitudes
enso_amplitudes = [random.uniform(0,20) for _ in range(10)]
enso_periods = [random.uniform(3, 7) for _ in range(10)]


##randomly selecting a percent of the time periods to throw gaps in
t_gap_frac = 0.10
max_nans = int(t_gap_frac*len(t))
num_nans = random.randint(0, max_nans)
nan_idxes = random.sample(range(len(t)), num_nans)

##Building matrix, right now it's just the same timeseries on all transects
for i in range(num_transects):
    for j in range(num_timesteps):
        ##Linear trend + six month cycle + yearly cycle + decadal cycle
        matrix[i,j] = sum([linear_trend(t[i], trend_val)*j,
                           sine_pattern(t[i], six_month_amplitude, 0.5)*j,
                           enso_pattern(t[i], enso_amplitudes, enso_periods)*j
                           ]
                          )
        ##Add random noise to each position
        matrix[i,j] = noise(matrix[i,j], noise_val)

##Apply nans to random timesteps for all transects
for i in range(num_timesteps):
    matrix[:,i] = apply_NANs(matrix[:,i], nan_idxes)
    
##Plot timeseries
plt.rcParams["figure.figsize"] = (12,4)
plt.plot(datetimes, matrix[:,0], '--o', c='k', linewidth=1, markersize=1)
plt.xlim(min(datetimes), max(datetimes))
plt.ylim(np.nanmin(matrix[:,0]), np.nanmax(matrix[:,0]))
plt.tight_layout()
plt.xlabel('Time (UTC)')
plt.ylabel('Cross-Shore Position (m)')
plt.savefig('timeseries.png')

##This is for plotting the matrix
y_lims = [datetimes[0], datetimes[-1]]
y_lims = mdates.date2num(y_lims)
plt.rcParams["figure.figsize"] = (8,8)
fig, ax = plt.subplots()
title = ('Linear Trend = ' +str(np.round(trend_val, 2)) +
         ' m/year\nDecadal Cycle Amplitude = ' +
         str(np.round(decadal_amplitude, 2))+
         ' m\n12 Month Cycle Amplitude = '+
         str(np.round(yearly_amplitude, 2)) +
         ' m\n6 Month Cycle Amplitude = ' +
         str(np.round(six_month_amplitude, 2))+
         ' m\nTemporal Frequency = '+str(dt)+' days'+
         '\nPositional Uncertainty = +/- '+str(noise_val) + ' m' +
         '\nPercentage of Missing Timesteps = '+str(t_gap_frac*100)+'%')
fig.suptitle(title)
im = ax.imshow(np.flipud(matrix),
          extent = [0, matrix.shape[1], y_lims[0], y_lims[1]],
          cmap='RdBu',
          aspect='auto')
ax.yaxis_date()
ax.set_ylabel('Time (UTC)')
ax.set_xlabel('Transect ID')
fig.colorbar(im, label='Cross-Shore Position (m)')
plt.tight_layout()
plt.savefig('matrix.png')
plt.close()







    
