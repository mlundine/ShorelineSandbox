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
import shoreline_timeseries_analysis_single as stsa
import os

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
                          datetime.timedelta(days=int(dt))
                          ).astype(datetime.datetime)
    num_transects = len(datetimes)
    shoreline_matrix = np.zeros((len(datetimes)))
    return shoreline_matrix, datetimes


def make_data(noise_val,
              trend_val,
              yearly_amplitude,
              dt,
              t_gap_frac,
              save_name):
    ##Initialize stuff
    #random.seed(0) #uncomment if you want to keep the randomness the same and play with hard-coded values
    matrix, datetimes = make_matrix(dt)
    num_timesteps = matrix.shape[0]
    t = time_array_to_years(datetimes)

    ##randomly selecting a percent of the time periods to throw gaps in
    max_nans = int(t_gap_frac*len(t))
    num_nans = random.randint(0, max_nans)
    nan_idxes = random.sample(range(len(t)), num_nans)

    ##Building matrix
    for i in range(num_timesteps):
         ##Linear trend + yearly cycle + noise
        matrix[i] = sum([linear_trend(t[i], trend_val),
                         sine_pattern(t[i], yearly_amplitude, 1),
                         noise(matrix[i], noise_val)
                         ]
                        )

    matrix = apply_NANs(matrix, nan_idxes)

    df = pd.DataFrame({'date':datetimes,
                       'position':matrix})
    df.to_csv(save_name+'.csv', index=False)

    plt.rcParams["figure.figsize"] = (16,4)
    plt.title(r'Trend value = ' +str(np.round(trend_val,2)) +
              r'm/year    Yearly Amplitude = ' +
              str(np.round(yearly_amplitude,2)) + r'm    dt = ' +
              str(dt) + r' days    Missing timestamps = ' +
              str(t_gap_frac*100)+'%')
    plt.plot(df['date'], df['position'], '--o', color='k', markersize=1, linewidth=1)
    plt.xlim(min(df['date']), max(df['date']))
    plt.ylim(min(df['position']), max(df['position']))
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_name+'.png', dpi=300)
    return save_name

random.seed(0)
##Noise value in meters
noise_val = 20
##Trend value in m/year
trend_val = random.uniform(-20, 20)
##Amplitude for yearly pattern in m
yearly_amplitude = random.uniform(0, 20)
##Revisit time in days
dt = 12
##Fraction of missing time periods
t_gap_frac = 0.15
##give it a save_name
save_name = 'test1'

make_data(noise_val,
          trend_val,
          yearly_amplitude,
          dt,
          t_gap_frac,
          save_name)


