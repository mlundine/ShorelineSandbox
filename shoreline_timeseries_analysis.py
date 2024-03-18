"""
Mark Lundine
This is an in-progress script with tools for processing 1D shoreline timeseries data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import random
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

def adf_test(timeseries):
    """
    Checks for stationarity (lack of trend) in timeseries
    I hate hypothesis testing!!!!!!!!!!!!
    significance value here is going to be 0.05
    If p-value > 0.05 then we are interpeting the tiemseries as stationary
    otherwise it's interpreted as non-stationary
    inputs:
    timeseries:
    outputs:
    stationary_bool: if p-value > 0.05, return True, if p-value < 0.05 return False
    """
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if df['p-value'] > 0.05:
        stationary_bool = True
    else:
        stationary_bool = False
    return stationary_bool

def get_linear_trend(df, trend_plot_path):
    """
    LLS on single transect timeseries
    inputs:
    df (pandas DataFrame): two columns, dates and cross-shore positions
    trend_plot_path (str): path to save plot to
    outputs:
    lls_result: all the lls results (slope, intercept, stderr, intercept_stderr, rvalue)
    x: datetimes in years
    """
    
    datetimes = np.array(df['date'])
    shore_pos = np.array(df['position'])
    datetimes_seconds = [None]*len(datetimes)
    initial_time = datetimes[0]
    for i in range(len(filter_df)):
        t = filter_df['dates'].iloc[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    x = datetimes_years
    y = shore_pos
    lls_result = stats.linregress(x,y)
    return lls_result, x

def de_trend_timeseries(df, lls_result, x):
    """
    de-trends the shoreline timeseries
    """
    slope = lls_result.slope
    intercept = lls_result.intercept
    y = df['position']
    
    fitx = np.linspace(min(x),max(x),len(x))
    fity = slope*fitx + intercept

    df['position'] = y - fit1y
    return df

def de_mean_timeseries(df):
    """
    de-means the shoreline timeseries
    """
    mean_pos = np.mean(df['position'])
    df['position'] = df['position']-mean_pos
    return df
    
def get_shoreline_data(csv_path):
    """
    Reads and reformats the timeseries into a pandas dataframe with datetime index
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    new_df = pd.DataFrame({'position':df['position']},
                          index=df['date'])
    return new_df

def compute_avg_and_max_time_delta(df):
    """
    Computes average and max time delta for timeseries rounded to days
    Need to drop the nan rows to compute these
    returns average and maximum timedeltas
    """
    df = df.dropna()
    datetimes = df.index
    timedeltas = [datetimes[i-1]-datetimes[i] for i in range(1, len(datetimes))]
    avg_timedelta = sum(timedeltas, datetime.timedelta(0)) / len(timedeltas)
    max_timedelta = max(timedeltas)
    return avg_timedelta.days, max_timedelta.days

def resample_timeseries(df, timedelta):
    """
    Resamples the timeseries according to the provided timedelta
    """
    new_df = df.resample(timedelta).mean()
    return new_df

def fill_nans(df):
    """
    Fills nans in timeseries with linear interpolation, keep it simple student
    """
    new_df = df.interpolate(method='linear')
    return new_df

def plot_autocorrelation(output_folder,
                         name,
                         df):
    """
    This computes and plots the autocorrelation
    Autocorrelation tells you how much a timeseris is correlated with a lagged version of itself.
    Useful for distinguishing timeseries with patterns vs random timeseries
    For example, for a timeseries sampled every 1 month,
    if you find a strong positive correlation (close to +1) at a lag of 12,
    then your timeseries is showing repeating yearly patterns.

    Alternatively, if it's a spatial series, sampled every 50m,
    if you find a strong negative correlation (close to -1) at a lag of 20,
    then you might interpret this as evidence for something like the presence of littoral drift.

    If the autocorrelation is zero for all lags, the series is random--good luck finding any meaning from it!
    """
    fig_save = os.path.join(output_folder, name+'autocorrelation.png')
    # Creating Autocorrelation plot
    x = pd.plotting.autocorrelation_plot(df['position'])
 
    # plotting the Curve
    x.plot()
 
    # Display
    plt.savefig(fig_save, dpi=300)
    plt.close()

def compute_approximate_entropy(U, m, r):
    """Compute Aproximate entropy, from https://en.wikipedia.org/wiki/Approximate_entropy
    If this value is high, then the timeseries is probably unpredictable.
    If this value is low, then the timeseries is probably predictable.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))
    
def make_plots(output_folder,
               name,
               approximate_entropy
               df,
               df_resampled,
               df_no_nans,
               df_de_meaned,
               df_de_trend=None):
    """
    Making timeseries plots of data, vertically stacked
    """
    fig_save = os.path.join(output_folder, name+'timeseries.png')
    
    if df_de_trend = None:
        plt.rcParams["figure.figsize"] = (16,12)
        plt.suptitle(Name + '\nApproximate Entropy = ' + str(np.round(approximate_entropy, 3)))
        ##Raw 
        plt.subplot(4,1,1)
        plt.plot(df.index, df['position'], '--o', color='k', label='Raw')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df['position']), np.nanmin(df['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##Resampled
        plt.subplot(4,1,2)
        plt.plot(df_resampled.index, df_resampled['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df_resampled.index), max(df_resampled.index))
        plt.ylim(np.nanmin(df_resampled['position']), np.nanmin(df_resampled['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##Interpolated
        plt.subplot(4,1,3)
        plt.plot(df_no_nans.index, df_no_nans['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df_no_nans.index), max(df_no_nans.index))
        plt.ylim(np.nanmin(df_no_nans['position']), np.nanmin(df_no_nans['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##De-meaned
        plt.subplot(4,1,4)
        plt.plot(df_de_meaned.index, df_de_meaned['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df_de_meaned.index), max(df_de_meaned.index))
        plt.ylim(np.nanmin(df_de_meaned['position']), np.nanmin(df_de_meaned['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlabel('Time (UTC)')
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_save, dpi=300)
        plt.close()
    else:
        plt.rcParams["figure.figsize"] = (16,15)
        plt.suptitle(Name + '\nApproximate Entropy = ' + str(np.round(approximate_entropy, 3)))
        ##Raw 
        plt.subplot(5,1,1)
        plt.plot(df.index, df['position'], '--o', color='k', label='Raw')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df['position']), np.nanmin(df['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##Resampled
        plt.subplot(5,1,2)
        plt.plot(df_resampled.index, df_resampled['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df_resampled.index), max(df_resampled.index))
        plt.ylim(np.nanmin(df_resampled['position']), np.nanmin(df_resampled['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##Interpolated
        plt.subplot(5,1,3)
        plt.plot(df_no_nans.index, df_no_nans['position'], '--o', color='k', label='Interpolated')
        plt.xlim(min(df_no_nans.index), max(df_no_nans.index))
        plt.ylim(np.nanmin(df_no_nans['position']), np.nanmin(df_no_nans['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##De-meaned
        plt.subplot(5,1,4)
        plt.plot(df_de_meaned.index, df_de_meaned['position'], '--o', color='k', label='De-Meaned')
        plt.xlim(min(df_de_meaned.index), max(df_de_meaned.index))
        plt.ylim(np.nanmin(df_de_meaned['position']), np.nanmin(df_de_meaned['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        ##De-Trended
        plt.subplot(5,1,4)
        plt.plot(df_de_trend.index, df_de_trend['position'], '--o', color='k', label='De-Meaned')
        plt.xlim(min(df_de_trend.index), max(df_de_trend.index))
        plt.ylim(np.nanmin(df_de_trend['position']), np.nanmin(df_de_trend['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlabel('Time (UTC)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_save, dpi=300)
        plt.close()
    
def main(csv_path,
         output_folder,
         name):
    """
    Timeseries analysis for satellite shoreline data
    inputs:
    csv_path (str): path to the shoreline timeseries csv
    should have columns 'date' and 'position'
    where date contains UTC datetimes in the format YYYY-mm-dd HH:MM:SS
    position is the cross-shore position of the shoreline
    output_folder (str): path to save outputs to
    """
    ##Step 1: Load in data
    df = get_shoreline_data(csv_path)
    
    ##Step 2: Compute average and max time delta
    avg_timedelta, max_timedelta = compute_avg_and_max_time_delta(df)

    ##Step 3: Resample timeseries to the maximum timedelta
    df_resampled = resample_timeseries(df, max_timedelta)

    ##Step 4: Fill NaNs
    df_no_nans = fill_nans(df_resampled)

    ##Step 5: De-mean the timeseries
    df_de_meaned = de_mean(df_no_nans)
    
    ##Step 6: Check for stationarity with ADF test
    stationary_bool = adf_test(df_de_meaned)
    
    ##Step 7a: If timeseries stationary, compute autocorrelation and approximate entropy
    ##Then make plots
    if stationary_bool == True:
        plot_autocorrelation(output_folder,
                             name,
                             df_de_meaned)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'],
                                                          2,
                                                          np.std(df_de_meaned['position']))
        make_plots(output_folder,
                   name,
                   approximate_entropy,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend=None)
        
    ##Step 7b: If timeseries non-stationary, compute trend, de-trend, compute autocorrelation and approximate entropy
    ##Then make plots
    else:
        trend_result, x = get_trend(df_de_meaned)
        df_de_trend = de_trend_timeseries(df_de_meaned, trend_result, x)
        plot_autocorrelation(output_folder,
                             name,
                             df_de_trend)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'],
                                                          2,
                                                          np.std(df_de_trend['position']))
        make_plots(output_folder,
                   name,
                   approximate_entropy,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend=df_de_trend)

    ##Step 8: Granger tests to see if other datasets are good predictors (e.g., waves, ENSO, NAO)
##    granger_result = grangercausalitytests(df['position'],
##                                           other_dataset,
##                                           maxlag=2)
        
    







    

