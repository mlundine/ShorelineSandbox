# ShorelineSandbox
Exploring the limits of satellite shorelines with synthetic data.

Can we see more in these datasets currently other than big trends?

Help wanted...

# Part 1: Processing Cookbook (time domain)

1. Resample timeseries to minimum, average, or maximum time delta (temporal spacing of timeseries). My gut is to go with the maximum so we aren't creating data. If we go with minimum or average then linearly interpolate the values to get rid of NaNs.

2. Check if timeseries is stationary with ADF test. We'll use a p-value of 0.05. If we get a p-value greater than this then we are interpreting
the timeseries as non-stationary (there is a temporal trend). 

If the timeseries is stationary then de-mean it, compute and plot autocorrelation, compute approximate entropy (measure of how predictable the timeseries is, values towards 0 indicate predictability, values towards 1 indicate random).

If the timeseries is non-stationary then compute the trend with linear least squares,
and then de-trend the timeseries. Then de-mean, do autocorrelation, and approximate entropy.

3. To-Do: Seasonal trend decomposition...

# Part 2: Processing Cookbook (spatial domain)

TO-DO

# Part 3:

Make a bunch of synthetic timeseries with known processes input.

See how much noise and data gaps we can add until any meaning is lost.

# Part 4:

Look at actual satellite-derived shoreline data and see where on the spectrum it falls. Try to identify necessary areas to improve.

# Part 5:

Similarly, make a bunch of synthetic timeseries with known processes input

See how much noise and data gaps we can add until an ML model fails at producing accurate predictions.

Compare with actual satellite derived shoreline data.





