"""
Mark Lundine
This script is in progress.
The goal is to take the CoastSeg matrix and resample it intelligently
in the time and space domain so that it is equally spaced temporally and spatially.
This will allow for predictive modeling from satellite shoreline data obtained from CoastSeg.
"""
import os
import shoreline_timeseries_analysis_single as stas
import shoreline_timeseries_analysis_single_spatial as stasp


##This is not functional at all yet
def main(transect_timeseries_path,
         config_gdf):
    """
    Performs timeseries and spatial series analysis cookbook on each
    transect in the transect_time_series matrix from CoastSeg
    inputs:
    transect_timeseries_path (str): path to the transect_time_series.csv
    config_gdf_path (str): path to the config_gdf.geojson
    """

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'],
                                              format='%Y-%m-%d %H:%M:%S+00:00')
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf[config_gdf['type']=='transect']
    
    timeseries_csvs = [None]*len(transects)
    transect_ids = [None]*len(transects)
    timeseries_mat_list = [None]*len(transects)
    for i in range(len(transects)):
        transect_id = transects['id'].iloc[i]
        timeseries_csv_path = os.path.join(timeseries_csv_dir, transect_id+'.csv')
        timeseries_plot_path = os.path.join(timeseries_plot_dir, transect_id+'_timeseries.png')
        
        dates = timeseries_data['dates']
        
        try:
            select_timeseries = np.array(timeseries_data[transect_id])
        except:
            i=i+1
            continue

        transect_ids[i] = transect_id
        
        ##Some simple timeseries processing
        data = pd.DataFrame({'distances':select_timeseries},
                            index=dates)
