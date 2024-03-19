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
                          datetime.timedelta(days=dt)
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
    t_gap_frac = 0.1
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
    df.to_csv(save_name, index=False)
    return save_name

def experiment(experiment_name,
               save_folder):
    random.seed(0)
    ##Making noise values in meters
    noise_vals = np.array([0, 5, 10, 20, 25, 30])
    ##Trend values in m/year
    trend_vals = np.array([-20, -15, -10, -5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5, 10, 15, 20])
    ##yearly_amplitude in meters
    yearly_amplitudes = np.array([0, 1, 2.5, 5, 10, 15, 20])
    ##revisit times in days
    dts = np.array([1, 4, 8, 12, 16, 30])
    ##fraction of missing times
    t_gap_fracs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    len_df = len(noise_vals)*len(trend_vals)*len(yearly_amplitudes)*len(dts)*len(t_gap_fracs)
    noise_vals_col = [None]*len_df
    trend_vals_col = [None]*len_df
    yearly_amplitudes_col = [None]*len_df
    dts_col = [None]*len_df
    t_gap_fracs_col = [None]*len_df
    computed_trends_col = [None]*len_df
    computed_trend_unc_col = [None]*len_df
    autocorr_max_col = [None]*len_df
    lag_max_col = [None]*len_df
    new_timedelta_col = [None]*len_df
    snr_no_nans_col = [None]*len_df
    approx_entropy_col = [None]*len_df
    save_names_col = [None]*len_df

    ##Make output_folder
    save_dir = os.path.join(save_folder, experiment_name)
    try:
        os.mkdir(save_dir)
    except:
        pass
    ##Yikes
    i=0
    for noise_val in noise_vals:
        for trend_val in trend_vals:
            for yearly_amplitude in yearly_amplitudes:
                for dt in dts:
                    for t_gap_frac in t_gap_fracs:
                        trial_name = experiment_name+'_'+str(i)
                        trial_dir = os.path.join(save_dir, trial_name)
                        synthetic_data_path = os.path.join(trial_dir, 'synthetic.csv')
                        try:
                            os.mkdir(trial_dir)
                        except:
                            pass
                        ##Make the synthetic data
                        synthetic_data = make_data(noise_val,
                                                   trend_val,
                                                   yearly_amplitude,
                                                   dt,
                                                   t_gap_frac,
                                                   synthetic_data_path)
                        ##Apply timeseries analysis
                        timeseries_analysis_result = stsa.main(synthetic_data,
                                                               trial_dir,
                                                               trial_name,
                                                               'maximum')
                        noise_vals_col[i] = noise_val
                        trend_vals_col[i] = trend_val
                        yearly_amplitudes_col[i] = yearly_amplitude
                        dts_col[i] = dt
                        t_gap_fracs_col[i] = t_gap_frac
                        computed_trends_col[i] = timeseries_analysis_result['computed_trend']
                        computed_trend_unc_col[i] = timeseries_analysis_result['trend_unc']
                        autocorr_max_col[i] = timeseries_analysis_result['autocorr_max']
                        lag_max_col[i] = timeseries_analysis_result['lag_max']
                        new_timedelta_col[i] = timeseries_analysis_result['new_timedelta']
                        snr_no_nans_col[i] = timeseries_analysis_result['snr_no_nans']
                        approx_entropy_col[i] = timeseries_analysis_result['approx_entropy']
                        save_names_col[i] = trial_dir
                        i=i+1
                        print('Progress: ' + str(100*(i/len_df))+'%')
                        
                        

    entropy_df = pd.DataFrame({'save_names':save_names_col,
                               'noise_val':noise_vals_col,
                               'trend_vals':trend_vals_col,
                               'yearly_amplitudes':yearly_amplitudes_col,
                               'dts':dts_col,
                               't_gap_fracs':t_gap_fracs_col,
                               'computed_trends':computed_trends_col,
                               'computed_trends_unc':computed_trends_unc_col,
                               'autocorr_max':autocorr_max_col,
                               'lag_max':lag_max_col,
                               'new_timedelta':new_timedelta_col,
                               'snr_no_nans':snr_no_nans_col,
                               'approx_entropy':approx_entropy_col})
    entropy_df.to_csv(os.path.join(save_folder, experiment_name+'.csv'))

experiment('Timeseries_experiment', r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Code\USGS\ShorelineSandbox\ShorelineSandbox\Experiment1')
    
