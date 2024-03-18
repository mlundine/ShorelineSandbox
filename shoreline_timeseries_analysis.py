"""
Mark Lundine
This is an in-progress script with tools for processing shoreline timeseries data.
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


def plot_power_spectrum(ps, freqs, save_path):
    """
    Plots power spectrum, y on log scale
    x-scale shows the periods because no one thinks in terms of frequencies
    This figure basically shows what periods explain most of the variability of the timeseries
    inputs:
    ps: power spectrum
    freqs: frequency bins
    """
    plt.plot(freqs, ps)
    ##this is in 1/months, so 10 years, 5 years, 2 years, 1 year, 8 months, 6 months, 3 months, 2 months
    freq_range = [1/120, 1/60, 1/24, 1/12, 1/8, 1/6, 1/3, 1/2]
    freq_range = np.array(freq_range)
    freq_labels = np.round(1/freq_range, 0)
    plt.xticks(freq_range, freq_labels)
    plt.xlabel('Period (Months)')
    lab = r'$\frac{m^2}{cycles/s}$'
    plt.ylabel(r'Spectral Density (' + lab + ')')
    plt.xlim(0, 1/2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def DFT(x, dt):
    """
    Discrete Fourier Transform
    not fast but here readibility is better than speed
    inputs:
    x: sequence values
    dt: timedelta
    returns:
    freqs: frequencies
    ps: the "power spectrum" (m^2/(cycles/s))
    """
    x = x
    N = len(x) #gets record length of x
    T_0 = N*dt # gets actual time length of data
    dft = [None]*int(N/2) # list to fill with dft values
    freqs = [None]*int(N/2) # list to fill with frequencies
    ## Outer loop over frequencies, range(start, stop) goes from start to stop-1
    for n in range(0, int(N/2)):
        dft_element = 0
        # inner loop over all data
        for k in range(0,N):
            m = np.exp((-2j * np.pi * k * n)/N)
            dft_element = dft_element + x[k]*m
        dft_element = dft_element * dt
        freq_element = n/T_0
        dft[n] = dft_element
        freqs[n] = freq_element
    ## change dft from list to np array
    dft = np.array(dft)
    ## compute power spectrum
    ps = 2*np.abs(dft**2)/T_0
    freqs = np.array(freqs)
    return freqs, ps

def mst_with_loess(df, dt, period):
    mstl = MSTL(
        df,
        periods=[24, 24 * 7],  # The periods and windows must be the same length and will correspond to one another.
        windows=[101, 101],  # Setting this large along with `seasonal_deg=0` will force the seasonality to be periodic.
        iterate=3,
        stl_kwargs={
                    "trend":1001, # Setting this large will force the trend to be smoother.
                    "seasonal_deg":0, # Means the seasonal smoother is fit with a moving average.
                   }
    )
    res = mstl.fit()
    ax = res.plot()
    
    ###Seasonal Trend Decomposition with Loess
    ###Might have screwed this up??
    periods_years = [0.5]
    periods_dt = [int(a*365/dt) for a in periods_years]

    ##Another fateful decision here, how to interpolate the timeseries???
    data = pd.DataFrame(data=matrix[:,0], index=datetimes).interpolate(method='linear')

    ##More thought needed here
    res = MSTL(data,
               #periods=periods_dt,
               iterate=3).fit()
    res.plot()
    plt.tight_layout()
    plt.savefig('stdwl.png')
    plt.close()

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

def make_plots(output_folder,
               name,
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
        plt.plot(df_de_meaned.index, df_de_meaned['position'], '--o', color='k', label='De-Meaned')
        plt.xlim(min(df_de_meaned.index), max(df_de_meaned.index))
        plt.ylim(np.nanmin(df_de_meaned['position']), np.nanmin(df_de_meaned['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlabel('Time (UTC)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_save, dpi=300)
        plt.close()        
    
def main(csv_path,
         name,
         output_folder):
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
    
    ##Step 7a: If timeseries stationary, perform DFT
    if stationary_bool == True:
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend=None)
        #freqs, ps = DFT(df_de_meaned)
        
    ##Step 7b: If timeseries non-stationary, compute trend, de-trend, then perform DFT
    else:
        trend_result, x = get_trend(df_de_meaned)
        df_de_trend = de_trend_timeseries(df_de_meaned, trend_result, x)
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend=df_de_trend)
        #freqs, ps = DFT(df_de_trend)
        
    ##Step 8: Plot power spectrum
    #plot_power_spectrum(freqs, ps)

    ##Step 9: For non-stationary timeseries with seasonality, perform seasonal trend decomposition with LOESS
    #decompose_signal(df_de_meaned)







    

