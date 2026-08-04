# mplnet_utils.py

import netCDF4 as nc
from netCDF4 import num2date
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

def load_mplnet_data(list_of_file_paths):
    """
    Loads and extracts data from MPLNET NetCDF files.

    Parameters:
    -----------
    list_of_file_paths : list
        List of file paths to the NetCDF files. The order of the files should be as follows:
        [extinction_file_path, depol_file_path, backscatter_file_path, lidar_ratio_file_path, 
         aod_file_path]
        Any of these file paths can be `None` if that specific data is not available.

    Returns:
    --------
    data_dict : dict
        Dictionary containing the extracted data. Missing data will be represented as `None`.
    """
    # Initialize the dictionary
    data_dict = {
        'total_extinction': None,
        'total_depol': None,
        'total_backscatter': None,
        'total_lidar_ratio': None,
        'altitudes': None,
        'latitude': None,
        'longitude': None,
        'time': None,
        'time_array': None,
        'time_grid': None,
        'altitudes_grid': None,
        'total_aod': None
    }

    # Load and process the extinction data (if available)
    if list_of_file_paths[0]:
        dataset_extinction = nc.Dataset(list_of_file_paths[0], mode='r')
        data_dict['total_extinction'] = dataset_extinction.variables['extinction'][0, :, :]
        data_dict['altitudes'] = dataset_extinction.variables['altitude'][0, :]
        data_dict['latitude'] = dataset_extinction.variables['latitude'][0]
        data_dict['longitude'] = dataset_extinction.variables['longitude'][0]
        
        # Convert Julian dates to datetime
        julian_time = dataset_extinction.variables['time'][:]  # Julian dates
        time_units = dataset_extinction.variables['time'].units  # "days since -4713-01-01 12:00:00 UTC"
        calendar = dataset_extinction.variables['time'].calendar  # "gregorian"
        time_objects = num2date(julian_time, units=time_units, calendar=calendar)
        time = [datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S.%f") for t in time_objects]
        data_dict['time'] = time
        data_dict['time_array'] = np.array(time)

        # Create time and altitude grids
        time_grid = np.repeat(data_dict['time_array'][:, np.newaxis], len(data_dict['altitudes']), axis=1)
        altitudes_grid = np.repeat(data_dict['altitudes'][np.newaxis, :], len(data_dict['time_array']), axis=0)
        data_dict['time_grid'] = time_grid
        data_dict['altitudes_grid'] = altitudes_grid

    # Load and process the depol data (if available)
    if list_of_file_paths[1]:
        dataset_depol = nc.Dataset(list_of_file_paths[1], mode='r')
        data_dict['total_depol'] = dataset_depol.variables['depol_ratio'][0, :, :]

    # Load and process the backscatter data (if available)
    if list_of_file_paths[2]:
        dataset_backscatter = nc.Dataset(list_of_file_paths[2], mode='r')
        data_dict['total_backscatter'] = dataset_backscatter.variables['backscatter'][0, :, :]

    # Load and process the lidar ratio data (if available)
    if list_of_file_paths[3]:
        dataset_lidar_ratio = nc.Dataset(list_of_file_paths[3], mode='r')
        data_dict['total_lidar_ratio'] = dataset_lidar_ratio.variables['lidar_ratio'][0, :]  # (wavelength, time)
        
    if list_of_file_paths[4]:
        dataset_aod = nc.Dataset(list_of_file_paths[4], mode='r')
        data_dict['total_aod'] = dataset_aod.variables['aod'][0,:] # (wavelength, time)

    return data_dict

def dust_fraction_calculation(total_depol, depol_dust, depol_nondust, total_backscatter, lidar_ratio_dust, total_extinction):
    """
    Calculates the dust fraction from the aerosol extinction profile.

    Parameters:
    -----------

    total_extinction : numpy.ndarray
        Particle aerosol extinction profile.
    total_backscatter : numpy.ndarray
        Particle aerosol backscatter profile.
    total_depol : numpy.ndarray
        Particle depolarization profile.
    depol_dust : float
        Depolarization ratio for dust particles.
    depol_nondust : float
        Depolarization ratio for non-dust particles.
    lidar_ratio_dust : float
        Lidar ratio for dust particles.

    Returns:
    --------
    dust_extinction : numpy.ndarray
        Dust extinction profile.
    dust_backscatter : numpy.ndarray
        Dust backscatter profile.
    """

    # Calculate the dust fraction
    depol_factor = np.ma.masked_array(total_depol, mask=np.ma.getmask(total_depol))
    depol_factor = ((total_depol - depol_nondust) *(1 + depol_nondust)) / ((depol_dust - depol_nondust)*(1+total_depol))
    depol_factor = np.clip(depol_factor, 0, 1)

    dust_backscatter = total_backscatter * depol_factor

    dust_extinction = lidar_ratio_dust * dust_backscatter

 
    return dust_extinction 


def filtering(dataset_extinction, dataset_backscatter, dataset_depol):
    """
    It filters data based on specific flags.

    Parameters:
    -----------

    dataset_extinction : netcdf4 object
        Dataset containing extinction data.
    dataset_backscatter : netcdf4 object
        Dataset containing backscatter data.
    dataset_depol : netcdf4 object
        Dataset containing depolarization ratio data.

    Returns:
    --------

    backscatter_filtered : numpy.ndarray
        Backscatter filtered.
    extinction_filtered : numpy.ndarray
        Extinction filtered.
    depol_filtered : numpy.ndarray
        Depolarization ratio filtered.
    mask_b : numpy.ndarray
        Mask of backscatter and extinction filtered. 
    mask_d : numpy.ndarray
        Mask of the depolarization filtered.

    """

    total_extinction = dataset_extinction.variables['extinction'][0, :, :]
    total_depol = dataset_depol.variables['depol_ratio'][0, :, :]
    total_backscatter = dataset_backscatter.variables['backscatter'][0, :, :]

    # Backsscatter flags 
    qa_backscatter = dataset_backscatter.variables['qa_backscatter'][0,:]
    flag_qa_nrb_backscatter = dataset_backscatter.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_backscatter = dataset_backscatter.variables['flag_cloud_screen'][0,:]
    flag_inversion_backscatter = dataset_backscatter.variables['flag_inversion'][0,:]
    flag_layers_backscatter = dataset_backscatter.variables['flag_layers'][0,:]
    flag_sunphotometer_backscatter = dataset_backscatter.variables['flag_sunphotometer'][0,:]
    flag_aod = dataset_backscatter.variables['flag_aod'][0,:]

    flag_cloud_screen_grid = np.repeat(flag_cloud_screen_backscatter[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid = np.repeat(flag_inversion_backscatter[:, np.newaxis], 400, axis = 1)
    flag_layers_grid = np.repeat(flag_layers_backscatter[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid = np.repeat(flag_sunphotometer_backscatter[:, np.newaxis], 400, axis = 1)
    flag_aod_grid = np.repeat(flag_aod[:, np.newaxis], 400, axis = 1)
    
    # PLDR flags 
    qa_depol = dataset_depol.variables['qa_depol_ratio'][0,:]
    flag_qa_nrb_depol = dataset_depol.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_depol = dataset_depol.variables['flag_cloud_screen'][0,:]
    flag_inversion_depol = dataset_depol.variables['flag_inversion'][0,:]
    flag_layers_depol = dataset_depol.variables['flag_layers'][0,:]
    flag_sunphotometer_depol = dataset_depol.variables['flag_sunphotometer'][0,:]
    flag_aod_depol = dataset_depol.variables['flag_aod'][0,:]

    flag_cloud_screen_grid_depol = np.repeat(flag_cloud_screen_depol[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid_depol = np.repeat(flag_inversion_depol[:, np.newaxis], 400, axis = 1)
    flag_layers_grid_depol = np.repeat(flag_layers_depol[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid_depol= np.repeat(flag_sunphotometer_depol[:, np.newaxis], 400, axis = 1)
    flag_aod_grid_depol = np.repeat(flag_aod_depol[:, np.newaxis], 400, axis = 1)

    # Filtering of backscatter and extinction (same mask)
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa = qa_backscatter >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen = flag_cloud_screen_grid > 1

    # Apply inversion filter
    mask_inversion = flag_inversion_grid > 1

    # Apply layers filtering
    mask_layers = flag_layers_grid > 1

    # Apply sunphotometer filter
    mask_sunphotometer = flag_sunphotometer_grid >= 8

    # Apply AOD filter
    mask_aod = flag_aod_grid >= 8

    # Mask negative values
    mask_neg = total_backscatter < 0
    
    # Mask NaN values
    mask_back_and_ext = np.isnan(total_backscatter)

    # mask_backscatter = np.logical_or(mask_qa, mask_cloud_screen)
    # mask_backscatter = np.logical_or(mask_backscatter, mask_inversion)
    # mask_backscatter = np.logical_or(mask_backscatter, mask_layers)
    # mask_backscatter = np.logical_or(mask_backscatter, mask_sunphotometer)
    # mask_backscatter = np.logical_or(mask_backscatter, mask_aod)
    # mask_backscatter = np.logical_or(mask_backscatter, mask_neg)
    
    mask_backscatter = np.logical_or.reduce([mask_qa, mask_cloud_screen, mask_inversion, mask_layers, mask_sunphotometer, mask_aod, mask_neg, mask_back_and_ext])

    new_backscatter = np.ma.array(total_backscatter, mask = mask_backscatter)
    new_extinction = np.ma.array(total_extinction, mask = mask_backscatter)

    # Filtering of PLDR
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa_depol = qa_depol >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen_depol = flag_cloud_screen_grid_depol > 1

    # Apply inversion filter
    mask_inversion_depol = flag_inversion_grid_depol > 1

    # Apply layers filtering
    mask_layers_depol = flag_layers_grid_depol > 1

    # Apply sunphotometer filter
    mask_sunphotometer_depol = flag_sunphotometer_grid_depol >= 8

    # Apply AOD filter
    mask_aod_depol = flag_aod_grid_depol >= 8

    # Mask values greater than 1
    mm = total_depol > 1
    
    # Masking NaN values 
    nan_mask_depol = np.isnan(total_depol)

    # mask_depol = np.logical_or(mask_qa_depol, mask_cloud_screen_depol)
    # mask_depol = np.logical_or(mask_depol, mask_inversion_depol)
    # mask_depol = np.logical_or(mask_depol, mask_layers_depol)
    # mask_depol = np.logical_or(mask_depol, mask_sunphotometer_depol)
    # mask_depol = np.logical_or(mask_depol, mask_aod_depol)
    # mask_depol = np.logical_or(mask_depol, mm)

    mask_depol = np.logical_or.reduce([mask_qa_depol, mask_cloud_screen_depol, mask_inversion_depol, mask_layers_depol, mask_sunphotometer_depol, mask_aod_depol, mm, nan_mask_depol])
    new_depol = np.ma.array(total_depol, mask = mask_depol)

    final_mask = np.logical_or(mask_depol, mask_backscatter)

    backscatter_filtered = np.ma.array(total_backscatter, mask = final_mask)
    extinction_filtered = np.ma.array(total_extinction, mask = final_mask)
    depol_filtered = np.ma.array(total_depol, mask = final_mask)
    
    mask_b = np.ma.getmask(backscatter_filtered)
    mask_d = np.ma.getmask(depol_filtered)

    return backscatter_filtered, extinction_filtered, depol_filtered, mask_b, mask_d


def mplnet_averaging(parameter, date):
    """
    It computes the 3-hour average of a specific variable. 

    Parameters:
    -----------

    parameter : numpy.ndarray
        Parameter to be averaged.
    date : string
        Date of the file under analysis (format %Y%m%d ex: 20240714)

    Returns:
    --------

    parameter_3hourly : numpy.ndarray
        Parameter averaged every 3 hours.
    time_3hourly : list of datetime objects
        List containing 8 times: 00, 03, 06, 09, 12, 15, 18, 21.

    """
    
    parameter_3hourly = []
    time_3hourly = []
    file_date = datetime.strptime(date, '%Y%m%d')

    parameter_blocks = parameter.reshape(8, 180, parameter.shape[1])

    for block_index2, block2 in enumerate(parameter_blocks):
        parameter_3hourly.append(np.ma.mean(block2, axis=0))
        time_3hourly.append(file_date + timedelta(hours=3*block_index2))

    parameter_3hourly = np.ma.array(parameter_3hourly)

    return parameter_3hourly, time_3hourly


def plot_extinction_vs_dust_concentration(s, k, total_extinction, dust_extinction, time_3hourly, altitudes, levels, dust_conc, lidar_ratio_dust):

    idx = 0

    for idx in range(len(s)):

        fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        if np.isnan(total_extinction[s[idx],:]).all():
            print(f' tot extinction is nan at time {time_3hourly[idx]}')
            r = s[idx]
            for l in range(1,58):
                if not np.isnan(total_extinction[r + l,:]).all():
                    new_time = r + l +1 
                    break
            ax[0].plot(total_extinction[new_time, :], altitudes, label='Total Extinction')
            ax[0].fill_betweenx(altitudes, total_extinction[new_time, :], 0, where=total_extinction[new_time, :] > 0, color='skyblue', alpha=0.5)
            ax[0].plot(dust_extinction[new_time,:], altitudes, color = 'red', linestyle='--', label='Dust Extinction')
            ax[0].fill_betweenx(altitudes, dust_extinction[new_time,:], 0, where=dust_extinction[new_time,:] > 0, color='red', alpha=0.5)      
        else: 
            ax[0].plot(total_extinction[s[idx],:], altitudes, label='Total Extinction')
            ax[0].fill_betweenx(altitudes, total_extinction[s[idx],:], 0, where=total_extinction[s[idx],:] > 0, color='skyblue', alpha=0.5)
            ax[0].plot(dust_extinction[s[idx],:], altitudes, color = 'red', linestyle='--', label='Dust Extinction')
            ax[0].fill_betweenx(altitudes, dust_extinction[s[idx],:], 0, where=dust_extinction[s[idx],:] > 0, color='red', alpha=0.5)
        ax[0].set_title(f'Dust Extinction on {time_3hourly[idx]} (dust lidar ratio = {lidar_ratio_dust})')
        ax[0].set_xlabel('Extinction (km-1)')
        ax[0].set_ylabel('Altitude (km)')
        ax[0].set_ylim(0, 10)
        ax[0].legend()
        ax[0].grid()


        ax[1].plot(dust_conc[idx,:]*10**9,levels/1000, label='Monarch Dust Concentration')
        ax[1].fill_betweenx(levels/1000, dust_conc[idx, :]*10**9, 0, where=dust_conc[idx, :]*10**9 > 0, color='skyblue', alpha=0.5)
        ax[1].set_title(f'Dust Concentration on {time_3hourly[idx]} (dust lidar ratio = {lidar_ratio_dust})')
        ax[1].set_xlabel('Dust Concentration (µg/m³)')
        ax[1].set_ylabel('Altitude (km)')
        ax[1].set_ylim(0, 10)
        ax[1].legend()
        ax[1].grid()


def plot_extinction_and_dust_concentration(s, dust_extinction_3hourly, time_3hourly, dust_conc, altitudes, levels, lidar_ratio_dust):

    for idx in range(len(s)):

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plotting dust extinction on ax1 (primary y-axis)
        ax1.plot(dust_extinction_3hourly[idx, :], altitudes, label='Dust Extinction 3-hourly', color='blue')
        ax1.fill_betweenx(altitudes, dust_extinction_3hourly[idx, :], 0, where=dust_extinction_3hourly[idx, :] > 0, color='skyblue', alpha=0.5)
        ax1.set_xlabel('Extinction (km⁻¹)', color='blue')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_ylim(0, 10)
        ax1.legend(loc='upper left')
        ax1.grid()
        ax1.tick_params(axis='x', labelcolor='blue')

        # Creating a twin axis for dust concentration (secondary x-axis)
        ax2 = ax1.twiny()

        # Plotting dust concentration on ax2
        ax2.plot(dust_conc[idx, :] * 10**9, levels / 1000, label='Monarch Dust Concentration', linestyle='--', color='red')
        ax2.fill_betweenx(levels / 1000, dust_conc[idx, :] * 10**9, 0, where=dust_conc[idx, :] * 10**9 > 0, color='lightcoral', alpha=0.5)
        ax2.set_xlabel('Dust Concentration (µg/m³)', color='red')
        ax2.set_ylim(0, 10)
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='x', labelcolor='red')

        # Setting axis limits for consistency
        ax1.set_xlim(left=0)  # Adjust based on your data ranges
        ax2.set_xlim(left=0)

        # Adding the title
        ax1.set_title(f'Dust Extinction and Dust Concentration on {time_3hourly[idx]} (dust lidar ratio = {lidar_ratio_dust})')

        # Show the plot
        plt.tight_layout()
        plt.show()
        
        
def extract_date_from_filename(filepath):
    """Extract the date from the filename (adjust this depending on your filename structure)"""
    # Assume filename structure MPLNET_V3_L15_AER_20240801_MPL44255_Santa_Cruz_Tenerife.nc4
    filename = filepath.stem  # Get the filename without extension
    date_str = filename.split('_')[4]  # Get the last part after underscore (YYYYMMDD)
    return datetime.strptime(date_str, '%Y%m%d')

        
#def save_to_csv2(time, parameter, parameter_name, directory_path): 
#        
#    df_parameter = pd.DataFrame(parameter)
#    df_parameter.insert(0, 'time', time)
#    
#    file_name = f'{parameter_name}.csv'
#    os.makedirs(directory_path, exist_ok=True)
#    output_file_path = os.path.join(directory_path, file_name)
#    df_parameter.to_csv(output_file_path, index=False)
#    print(f"CSV file saved to {output_file_path}")

              
def creating_mask(dataset_extinction, dataset_backscatter, dataset_depol):
    """
    It filters data based on specific flags.

    Parameters:
    -----------

    dataset_extinction : netcdf4 object
        Dataset containing extinction data.
    dataset_backscatter : netcdf4 object
        Dataset containing backscatter data.
    dataset_depol : netcdf4 object
        Dataset containing depolarization ratio data.

    Returns:
    --------

    final_mask : numpy.ndarray
        Mask to be applied to extinction, backscatter and depolarization. 

    """
    
    total_extinction = dataset_extinction.variables['extinction'][0, :, :]
    total_depol = dataset_depol.variables['depol_ratio'][0, :, :]
    total_backscatter = dataset_backscatter.variables['backscatter'][0, :, :]

    # Backsscatter flags 
    qa_backscatter = dataset_backscatter.variables['qa_backscatter'][0,:]
    flag_qa_nrb_backscatter = dataset_backscatter.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_backscatter = dataset_backscatter.variables['flag_cloud_screen'][0,:]
    flag_inversion_backscatter = dataset_backscatter.variables['flag_inversion'][0,:]
    flag_layers_backscatter = dataset_backscatter.variables['flag_layers'][0,:]
    flag_sunphotometer_backscatter = dataset_backscatter.variables['flag_sunphotometer'][0,:]
    flag_aod = dataset_backscatter.variables['flag_aod'][0,:]

    flag_cloud_screen_grid = np.repeat(flag_cloud_screen_backscatter[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid = np.repeat(flag_inversion_backscatter[:, np.newaxis], 400, axis = 1)
    flag_layers_grid = np.repeat(flag_layers_backscatter[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid = np.repeat(flag_sunphotometer_backscatter[:, np.newaxis], 400, axis = 1)
    flag_aod_grid = np.repeat(flag_aod[:, np.newaxis], 400, axis = 1)
    
    # PLDR flags 
    qa_depol = dataset_depol.variables['qa_depol_ratio'][0,:]
    flag_qa_nrb_depol = dataset_depol.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_depol = dataset_depol.variables['flag_cloud_screen'][0,:]
    flag_inversion_depol = dataset_depol.variables['flag_inversion'][0,:]
    flag_layers_depol = dataset_depol.variables['flag_layers'][0,:]
    flag_sunphotometer_depol = dataset_depol.variables['flag_sunphotometer'][0,:]
    flag_aod_depol = dataset_depol.variables['flag_aod'][0,:]

    flag_cloud_screen_grid_depol = np.repeat(flag_cloud_screen_depol[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid_depol = np.repeat(flag_inversion_depol[:, np.newaxis], 400, axis = 1)
    flag_layers_grid_depol = np.repeat(flag_layers_depol[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid_depol= np.repeat(flag_sunphotometer_depol[:, np.newaxis], 400, axis = 1)
    flag_aod_grid_depol = np.repeat(flag_aod_depol[:, np.newaxis], 400, axis = 1)

    # Filtering of backscatter and extinction (same mask)
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa = qa_backscatter >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen = flag_cloud_screen_grid > 1

    # Apply inversion filter
    mask_inversion = flag_inversion_grid > 1

    # Apply layers filtering
    mask_layers = flag_layers_grid > 1

    # Apply sunphotometer filter
    mask_sunphotometer = flag_sunphotometer_grid >= 8

    # Apply AOD filter
    mask_aod = flag_aod_grid >= 8
    
    flag_mask_back = np.logical_or.reduce([mask_qa, mask_cloud_screen, mask_inversion, mask_layers, mask_sunphotometer, mask_aod])
    # print('Number of samples masked thorugh flags (backscatter): ', np.sum(flag_mask_back))

    # Mask negative values
    mask_neg = total_backscatter < 0
    # print('Number of negative samples in backscatter and extinction: ', np.sum(mask_neg))
    
    # Mask NaN values
    mask_back_and_ext = np.isnan(total_backscatter)
    # print('Number of NaN samples in axtinction and backscatter ', np.sum(mask_back_and_ext))
    
    mask_backscatter = np.logical_or.reduce([mask_qa, mask_cloud_screen, mask_inversion, mask_layers, mask_sunphotometer, mask_aod, mask_neg, mask_back_and_ext])

    # Filtering of PLDR
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa_depol = qa_depol >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen_depol = flag_cloud_screen_grid_depol > 1

    # Apply inversion filter
    mask_inversion_depol = flag_inversion_grid_depol > 1

    # Apply layers filtering
    mask_layers_depol = flag_layers_grid_depol > 1

    # Apply sunphotometer filter
    mask_sunphotometer_depol = flag_sunphotometer_grid_depol >= 8

    # Apply AOD filter
    mask_aod_depol = flag_aod_grid_depol >= 8
    
    flag_mask_depol = np.logical_or.reduce([mask_qa_depol, mask_cloud_screen_depol, mask_inversion_depol, mask_layers_depol, mask_sunphotometer_depol, mask_aod_depol])
    # print('Number of samples masked through flags (depol): ', np.sum(flag_mask_depol))

    # Mask values greater than 1
    depol_greater_1 = total_depol > 1
    # print('Number of depol samples > 1 ', np.sum(depol_greater_1))
    
    # Masking NaN values 
    nan_mask_depol = np.isnan(total_depol)
    # print('Number of NaN values in depol: ', np.sum(nan_mask_depol))

    mask_depol = np.logical_or.reduce([mask_qa_depol, mask_cloud_screen_depol, mask_inversion_depol, mask_layers_depol, mask_sunphotometer_depol, mask_aod_depol, depol_greater_1, nan_mask_depol])

    final_mask = np.logical_or(mask_depol, mask_backscatter)

    return final_mask, flag_mask_back, mask_neg, mask_back_and_ext, flag_mask_depol, depol_greater_1, nan_mask_depol


def filtering2(dataset_extinction, dataset_backscatter, dataset_depol):
    """
    It filters data based on specific flags.

    Parameters:
    -----------

    dataset_extinction : netcdf4 object
        Dataset containing extinction data.
    dataset_backscatter : netcdf4 object
        Dataset containing backscatter data.
    dataset_depol : netcdf4 object
        Dataset containing depolarization ratio data.

    Returns:
    --------

    backscatter_filtered : numpy.ndarray
        Backscatter filtered.
    extinction_filtered : numpy.ndarray
        Extinction filtered.
    depol_filtered : numpy.ndarray
        Depolarization ratio filtered.
    mask_b : numpy.ndarray
        Mask of backscatter and extinction filtered. 
    mask_d : numpy.ndarray
        Mask of the depolarization filtered.

    """

    total_extinction = dataset_extinction.variables['extinction'][0, :, :]
    total_depol = dataset_depol.variables['depol_ratio'][0, :, :]
    total_backscatter = dataset_backscatter.variables['backscatter'][0, :, :]

    # Backsscatter flags 
    qa_backscatter = dataset_backscatter.variables['qa_backscatter'][0,:]
    flag_qa_nrb_backscatter = dataset_backscatter.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_backscatter = dataset_backscatter.variables['flag_cloud_screen'][0,:]
    flag_inversion_backscatter = dataset_backscatter.variables['flag_inversion'][0,:]
    flag_layers_backscatter = dataset_backscatter.variables['flag_layers'][0,:]
    flag_sunphotometer_backscatter = dataset_backscatter.variables['flag_sunphotometer'][0,:]
    flag_aod = dataset_backscatter.variables['flag_aod'][0,:]

    flag_cloud_screen_grid = np.repeat(flag_cloud_screen_backscatter[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid = np.repeat(flag_inversion_backscatter[:, np.newaxis], 400, axis = 1)
    flag_layers_grid = np.repeat(flag_layers_backscatter[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid = np.repeat(flag_sunphotometer_backscatter[:, np.newaxis], 400, axis = 1)
    flag_aod_grid = np.repeat(flag_aod[:, np.newaxis], 400, axis = 1)
    
    # PLDR flags 
    qa_depol = dataset_depol.variables['qa_depol_ratio'][0,:]
    flag_qa_nrb_depol = dataset_depol.variables['flag_qa_nrb'][:] # don't need this because there are only 1s
    flag_cloud_screen_depol = dataset_depol.variables['flag_cloud_screen'][0,:]
    flag_inversion_depol = dataset_depol.variables['flag_inversion'][0,:]
    flag_layers_depol = dataset_depol.variables['flag_layers'][0,:]
    flag_sunphotometer_depol = dataset_depol.variables['flag_sunphotometer'][0,:]
    flag_aod_depol = dataset_depol.variables['flag_aod'][0,:]

    flag_cloud_screen_grid_depol = np.repeat(flag_cloud_screen_depol[:, np.newaxis], 400, axis = 1)
    flag_inversion_grid_depol = np.repeat(flag_inversion_depol[:, np.newaxis], 400, axis = 1)
    flag_layers_grid_depol = np.repeat(flag_layers_depol[:, np.newaxis], 400, axis = 1)
    flag_sunphotometer_grid_depol= np.repeat(flag_sunphotometer_depol[:, np.newaxis], 400, axis = 1)
    flag_aod_grid_depol = np.repeat(flag_aod_depol[:, np.newaxis], 400, axis = 1)

    # Filtering of backscatter and extinction (same mask)
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa = qa_backscatter >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen = flag_cloud_screen_grid > 1

    # Apply inversion filter
    mask_inversion = flag_inversion_grid > 1

    # Apply layers filtering
    mask_layers = flag_layers_grid > 1

    # Apply sunphotometer filter
    mask_sunphotometer = flag_sunphotometer_grid >= 8

    # Apply AOD filter
    mask_aod = flag_aod_grid >= 8

    # Mask negative values
    mask_neg = total_backscatter < 0
    
    # Mask NaN values 
    mask_back_and_ext = np.isnan(total_backscatter)

    mask_backscatter = np.logical_or(mask_qa, mask_cloud_screen)
    mask_backscatter = np.logical_or(mask_backscatter, mask_inversion)
    mask_backscatter = np.logical_or(mask_backscatter, mask_layers)
    mask_backscatter = np.logical_or(mask_backscatter, mask_sunphotometer)
    mask_backscatter = np.logical_or(mask_backscatter, mask_aod)
    mask_backscatter = np.logical_or(mask_backscatter, mask_neg)
    mask_backscatter = np.logical_or(mask_backscatter, mask_back_and_ext)

    new_backscatter = np.ma.array(total_backscatter, mask = mask_backscatter)
    new_extinction = np.ma.array(total_extinction, mask = mask_backscatter)

    # Filtering of PLDR
    # Apply QA backscatter filter (QA >= 4, high and moderate quality)
    mask_qa_depol = qa_depol >= 8

    # Apply cloud screen filter (flag_cloud_screen > 1, clouds free)
    mask_cloud_screen_depol = flag_cloud_screen_grid_depol > 1

    # Apply inversion filter
    mask_inversion_depol = flag_inversion_grid_depol > 1

    # Apply layers filtering
    mask_layers_depol = flag_layers_grid_depol > 1

    # Apply sunphotometer filter
    mask_sunphotometer_depol = flag_sunphotometer_grid_depol >= 8

    # Apply AOD filter
    mask_aod_depol = flag_aod_grid_depol >= 8

    # Mask values greater than 1
    mm = total_depol > 1


    # Mask NaN values 
    nan_mask_depol = np.isnan(total_depol)
    
    mask_depol = np.logical_or(mask_qa_depol, mask_cloud_screen_depol)
    mask_depol = np.logical_or(mask_depol, mask_inversion_depol)
    mask_depol = np.logical_or(mask_depol, mask_layers_depol)
    mask_depol = np.logical_or(mask_depol, mask_sunphotometer_depol)
    mask_depol = np.logical_or(mask_depol, mask_aod_depol)
    mask_depol = np.logical_or(mask_depol, mm)
    mask_depol = np.logical_or(mask_depol, nan_mask_depol)

    new_depol = np.ma.array(total_depol, mask = mask_depol)

    
    final_mask = np.logical_or(mask_depol, mask_backscatter)
    
    backscatter_filtered = np.ma.array(total_backscatter, mask = final_mask)
    extinction_filtered = np.ma.array(total_extinction, mask = final_mask)
    depol_filtered = np.ma.array(total_depol, mask = final_mask)
    
    mask_b = np.ma.getmask(backscatter_filtered)
    mask_d = np.ma.getmask(depol_filtered)

    return backscatter_filtered, extinction_filtered, depol_filtered, mask_b, mask_d


def save_to_netcdf(time_array, dust_conc, levels, output_filename):
    # Define the reference time based on the first sample in `time_array`
    reference_time = time_array[0]
    ref_time_str = reference_time.strftime("%Y-%m-%d %H:%M:%S")
    time_units = f"hours since {ref_time_str}"

    with nc.Dataset(output_filename, "w", format="NETCDF4") as ds:
        # Create dimensions
        time_dim = ds.createDimension("time", len(time_array))
        lev_dim = ds.createDimension("lev", len(levels))
        
        # Create variables
        times = ds.createVariable("time", "f8", ("time",))
        levels_var = ds.createVariable("lev", "f4", ("lev",))
        dust_conc_var = ds.createVariable("dust_concentration", "f4", ("time", "lev"), fill_value=-999)
        
        # Assign data to variables
        times.units = time_units
        times.calendar = "standard"
        times[:] = nc.date2num(time_array, units=time_units, calendar=times.calendar)
        
        levels_var.units = "meters"
        levels_var[:] = levels
        
        dust_conc_var.units = "kg/m^3"
        dust_conc_var[:, :] = dust_conc
        
        # Add global attributes
        ds.description = f"Dust concentration at various altitude levels for {output_filename.name}"
        ds.history = "Created " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.source = "MONARCH model output"

    print(f"NetCDF file saved to {output_filename}")
    
def save_to_netcdf2(time_array, dust_load, output_filename):
    # Define the reference time based on the first sample in `time_array`
    reference_time = time_array[0]
    ref_time_str = reference_time.strftime("%Y-%m-%d %H:%M:%S")
    time_units = f"hours since {ref_time_str}"

    with nc.Dataset(output_filename, "w", format="NETCDF4") as ds:
        # Create dimensions
        time_dim = ds.createDimension("time", len(time_array))
        
        # Create variables
        times = ds.createVariable("time", "f8", ("time",))
        dust_load_var = ds.createVariable("dust_load", "f4", ("time",), fill_value=-999)
        
        # Assign data to variables
        times.units = time_units
        times.calendar = "standard"
        times[:] = nc.date2num(time_array, units=time_units, calendar=times.calendar)
        
        dust_load_var.units = "kg/m^2"
        dust_load_var[:] = dust_load
        
        # Add global attributes
        ds.description = f"Total dust load for {output_filename.name}"
        ds.history = "Created " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.source = "MONARCH model output"

    print(f"NetCDF file saved to {output_filename}")

def daily_average(parameter_df, avg_type):
    # Create a true copy of the DataFrame to avoid modifying the original
    parameter_copy = parameter_df.copy()
    
    parameter_copy['time'] = pd.to_datetime(parameter_copy['time'])
    parameter_copy.set_index('time', inplace=True)
    
    if avg_type == 'from 3 to 21':
        filtered_parameter_df = parameter_copy.between_time("03:00", "21:01")
        daily_average_df = filtered_parameter_df.resample('D').mean() 
        
    elif avg_type == 'all day':
        daily_average_df = parameter_copy.resample('D').mean()
        
    return daily_average_df


def correlation_across_days_per_level(mplnet_array, monarch_array, monarch_altitude_levels, mplnet_altitude_levels):
    
    mplnet_adapted_to_monarch = np.empty((mplnet_array.shape[0], len(monarch_altitude_levels)))

    # Loop over each MONARCH altitude level and find corresponding MPLNET range
    for j, monarch_altitude in enumerate(monarch_altitude_levels):
        if j == 0:
            lower_bound = 250
        else:
            lower_bound = (monarch_altitude_levels[j - 1] + monarch_altitude) / 2
        if j == len(monarch_altitude_levels) - 1:
            upper_bound = mplnet_altitude_levels[-1]
        else:
            upper_bound = (monarch_altitude + monarch_altitude_levels[j + 1]) / 2

        indices = np.where((mplnet_altitude_levels >= lower_bound) & (mplnet_altitude_levels < upper_bound))[0]

        if len(indices) > 0: 
            mplnet_adapted_to_monarch[:, j] = np.nanmean(mplnet_array[:, indices], axis=1)
        else:
            mplnet_adapted_to_monarch[:, j] = np.nan

    correlation_filtered = []

    # Calculate Pearson correlation for each MONARCH altitude level
    for level in range(mplnet_adapted_to_monarch.shape[1]):
        mplnet_filtered_data = mplnet_adapted_to_monarch[:, level]
        monarch_data = monarch_array[:, level] * 10**(-3)

        # Remove NaN or inf values
        valid_indices_filtered = np.isfinite(mplnet_filtered_data) & np.isfinite(monarch_data)
        
        if valid_indices_filtered.sum() > 1:  # At least two points
            corr, _ = pearsonr(mplnet_filtered_data[valid_indices_filtered], monarch_data[valid_indices_filtered])
            correlation_filtered.append(corr)
        else:
            correlation_filtered.append(np.nan)
    
    return correlation_filtered

def daily_correlation_across_all_levels(mplnet_array, monarch_array, monarch_altitude_levels, mplnet_altitude_levels, time):
    
    mplnet_adapted_to_monarch = np.empty((mplnet_array.shape[0], len(monarch_altitude_levels)))

    # Loop over each MONARCH altitude level and find corresponding MPLNET range
    for j, monarch_altitude in enumerate(monarch_altitude_levels):
        if j == 0:
            lower_bound = 250
        else:
            lower_bound = (monarch_altitude_levels[j - 1] + monarch_altitude) / 2
        if j == len(monarch_altitude_levels) - 1:
            upper_bound = mplnet_altitude_levels[-1]
        else:
            upper_bound = (monarch_altitude + monarch_altitude_levels[j + 1]) / 2

        indices = np.where((mplnet_altitude_levels >= lower_bound) & (mplnet_altitude_levels < upper_bound))[0]

        if len(indices) > 0: 
            mplnet_adapted_to_monarch[:, j] = np.nanmean(mplnet_array[:, indices], axis=1)
        else:
            mplnet_adapted_to_monarch[:, j] = np.nan
            
    
    daily_correlations = []
    for i in range(mplnet_adapted_to_monarch.shape[0]):
        # Remove NaN values to ensure pearsonr calculation
        valid_idx = ~np.isnan(mplnet_adapted_to_monarch[i]) & ~np.isnan(monarch_array[i])
        if np.sum(valid_idx) > 1:  # Ensure there are enough data points to calculate correlation
            correlation, _ = pearsonr(mplnet_adapted_to_monarch[i, valid_idx], monarch_array[i, valid_idx]*10**(-3))
        else:
            correlation = np.nan
        daily_correlations.append(correlation)

    # Convert daily correlations to a pandas DataFrame for better readability
    dates = pd.to_datetime(time)  # assuming time_july holds datetime-like objects
    correlation_df = pd.DataFrame({'Date': dates, 'Correlation': daily_correlations})
    
    return correlation_df


def daily_correlation_across_all_levels_no_adaptation(mplnet_array, monarch_array, time):
    
    daily_correlations = []
    for i in range(mplnet_array.shape[0]):
        valid_idx = ~np.isnan(mplnet_array[i]) & ~np.isnan(monarch_array[i])
        if np.sum(valid_idx) > 1:  # Ensure there are enough data points to calculate correlation
            correlation, _ = pearsonr(mplnet_array[i, valid_idx], monarch_array[i, valid_idx]*10**(-3))
        else:
            correlation = np.nan
        daily_correlations.append(correlation)
    
    dates = pd.to_datetime(time)  # assuming time_july holds datetime-like objects
    correlation_df = pd.DataFrame({'Date': dates, 'Correlation': daily_correlations})
    
    return correlation_df

    
    
    
    
