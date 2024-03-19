# ShorelineSandbox
Exploring the limits of satellite shorelines with synthetic data.

Can we see more in these datasets currently other than big trends?

Help wanted...

Goal: Identify areas where improvements in satellite derived shoreline data are needed to capture processes and build better predictive models.

Questions: 

1. Can we define a set of well-thought-out analysis tools that treat these datasets rigorously? 

2. Given the current state of satellite derived shoreline data, what is the smallest scale (temporally and spatially) that we can investigate?

3. Can a data-driven model be of any use for current satellite shoreline data?

4. Can we identify areas to improve these datasets?

# Part 1: Processing Cookbook (time domain)

Use shoreline_timeseries_analysis_single.py

Libraries required (Python 3.7, numpy, matplotlib, datetime, random, scipy, pandas, statsmodels, os, csv)

	main(csv_path,
             output_folder,
             name,
             which_timedelta):
    	"""
    	Timeseries analysis for satellite shoreline data.
	
	Will save timeseries plot (raw, resampled, de-trended, de-meaned) and autocorrelation plot.
	
	Will also output analysis results to a csv (result.csv).

    	inputs:
    	csv_path (str): path to the shoreline timeseries csv
    	should have columns 'date' and 'position'
    	where date contains UTC datetimes in the format YYYY-mm-dd HH:MM:SS
    	position is the cross-shore position of the shoreline
    	output_folder (str): path to save outputs to
    	name (str): name to give this analysis run
    	which_timedelta (str): 'minimum' 'average' or 'maximum', this is what the timeseries is resampled at
    	outputs:
    	timeseries_analysis_result (dict): results of this cookbook
	"""
1. Resample timeseries to minimum, average, or maximum time delta (temporal spacing of timeseries). My gut is to go with the maximum so we aren't creating data. If we go with minimum or average then linearly interpolate the values to get rid of NaNs.

2. Check if timeseries is stationary with ADF test. We'll use a p-value of 0.05. If we get a p-value greater than this then we are interpreting
the timeseries as non-stationary (there is a temporal trend). 

3. 
	a) If the timeseries is stationary then de-mean it, compute and plot autocorrelation, compute approximate entropy (measure of how predictable 		the timeseries is, values towards 0 indicate predictability, values towards 1 indicate random).

	b) If the timeseries is non-stationary then compute the trend with linear least squares,
	and then de-trend the timeseries. Then de-mean, do autocorrelation, and approximate entropy.

4. This will return a dictionary with the following keys:

	* 'stationary_bool': True or False, whether or not the input timeseries was stationary according to the ADF test.
	* 'computed_trend': a computed linear trend via linear least squares, m/year
	* 'computed_intercept': the computed intercept via linear least squares, m
	* 'trend_unc': the standard error of the linear trend estimate, m/year
	* 'intercept_unc': the staard of error of the intercept estimate, m
	* 'r_sq': the r^2 value from the linear trend estimation, unitless
	* 'autocorr_max': the maximum value from the autocorrelation estimation, this code computes the maximum of the absolute value of the 		autocorrelation
	* 'lag_max': the lag that corresponds to the maximum of the autocorrelation, something of note here: if you are computing autocorrelation on
	a signal with a period of 1 year, then here the lag_max will be half a year. Autocorrelation in this case should be -1 at a half-year lag and 		+1 at a year lag. Since I do the max calculation on the absolute value fo the autocorrelation, you get lag_max at the maximum negative 		correlation.
	* 'new_timedelta': this is the new time-spacing for the resampled timeseries
	* 'snr_no_nans': a crude estimate of signal-to-noise ratio, here I just did the mean of the timeseries divided by the standard deviation
	* 'approx_entropy': entropy estimate, values closer to 0 indicate predicatibility, values closer to 1 indicate disorder


# Part 2: Processing Cookbook (spatial domain)

Use shoreline_timeseries_analysis_single_spatial.py

Libraries required (Python 3.7, numpy, matplotlib, datetime, random, scipy, pandas, statsmodels, os, csv)

	main(csv_path,
	     output_folder,
             name,
             transect_spacing,
             which_spacedelta):
	"""
    	Spatial series analysis for satellite shoreline data.
	
	Will save spatial series plot (raw, resampled, de-trended, de-meaned) and autocorrelation plot.
	
	Will also output analysis results to a csv (result.csv).

    	inputs:
    	csv_path (str): path to the shoreline timeseries csv
    	should have columns 'transect_id' and 'position'
    	where transect_id contains the transect id, transects should be evenly spaced!!
    	position is the cross-shore position of the shoreline (in m)
    	output_folder (str): path to save outputs to
    	name (str): a site name
    	transect_spacing (int): transect spacing in meters
    	which_spacedelta (str): 'minimum' 'average' or 'maximum', this is the new longshore spacing to sample at
	outputs:
    	spatial_series_analysis_result (dict): results of this cookbook
    	"""

# Part 3:

Make a bunch of synthetic timeseries with known processes input.

See how much noise and data gaps we can add until any meaning is lost.

# Part 4:

Look at actual satellite-derived shoreline data and see where on the spectrum it falls. Try to identify necessary areas to improve.

Some things we should estimate: what is the average amount of missing timestamps in SDS data? How does this vary through time?

# Part 5:

Similarly, make a bunch of synthetic timeseries with known processes input

See how much noise and data gaps we can add until an ML model fails at producing accurate predictions.

Compare with actual satellite derived shoreline data.






