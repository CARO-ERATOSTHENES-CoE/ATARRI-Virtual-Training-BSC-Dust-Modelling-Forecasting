import pandas as pd
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset as nc
from io import StringIO
from datetime import datetime
import os.path
import argparse
import sys
from glob import glob

STATIONS = ["Qena_SVU"]

#### Main Formatting class for Aeronet data
class FormatAERONETv3(object):
    """ Class to format Aeronet v3 """

    def __init__(self, date=None, frequency='3H', level='15'):
        """ Initialize variables """
        self.date = date
        self.frequency = frequency
        self.level = level
        self.st_lons = 'Longitude(decimal_degrees)'
        self.st_lats = 'Latitude(decimal_degrees)'
        self.st_alts = 'Elevation(meters)'
        self.st_file = '../scripts/aeronet_locations_v3_daily.txt' ## ADD PATH HERE
        self.station = 5000
        self.fillval = -9999.
        self.nullvalue = -100.
        self.ae440_870aero = None
        self.od550aero = None
        self.lon = None
        self.lat = None
        self.alt = None
        self.time = None
        self.ndata = None
        self.station_name = None
        print(self.frequency)
        print(self.date)
        self.timesteps = pd.date_range('%s 00:00' % self.date,
                                       '%s 23:00' % self.date,
                                       freq=self.frequency)
        self.base = -(float(self.frequency[:-1])/2)
        self.period = self.frequency[-1]
        self.loffset = "{}{}".format(abs(self.base), self.period)
        self.t_steps = len(self.timesteps)


    def _nc_dims(self):
        return {
            'time': None,
            'station': 5000, #self.station,
            'strlen': 80,
        }


    def _st_vars(self):
        """
        All dependent variables
        """
        return {
            'lon': {
                'dims': ('station',),
                'dtype': 'f',
                'attrs': {
                    'standard_name': "longitude",
                    'units': "degrees_east",
                },
                'val': self.lon,
            },
            'lat': {
                'dims': ('station',),
                'dtype': 'f',
                'attrs': {
                    'standard_name': "latitude",
                    'units': "degrees_north",
                },
                'val': self.lat,
            },
            'alt': {
                'dims': ('station',),
                'dtype': 'f',
                'attrs': {
                    'standard_name': "altitude",
                    'units': "meters",
                },
                'val': self.alt,
            },
            'time': {
                'dims': ('time',),
                'dtype': 'f',
                'attrs': {
                    "standard_name":"time",
                    'units': "hours since %s",
                    'calendar': "gregorian",
                },
                'val': self.time,
            },
            'station': {
                'dims': ('station',),
                'dtype': 'f',
                'attrs': {
                    'long_name': 'number of available AERONET sites',
                    'units': '-',
                },
                'val': self.station_count,
            },
        }


    def _cm_vars(self):
        """
        Coordinate variables
        """
        return {
            'ndata': {
                'dims': ('time', 'station'),
                'dtype': 'i',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "number of observations used to calculate the value",
                },
                'val': self.ndata,
            },
            'station_name': {
                'dims': ('station', 'strlen'),
                'dtype': 'c',
                'attrs': {
                    'units': '-',
                    'long_name': "station name",
                },
                'val': self.station_name_full,
            },
        }

    def write_monthly_var(self, mon, ncvar, outdir):
        """ 
        Write NetCDF4 zipped files according to standards 
        """

        nc_dims = self._nc_dims()
        nc_vars = self._nc_vars()
        cm_vars = self._cm_vars()
        cm_vars.update(self._st_vars())
        
        # set the first day of following month
        mon1 = (pd.to_datetime(mon, format='%Y%m') + pd.Timedelta(days=31)).strftime('%Y%m')

        # monthly timesteps
        m_timesteps = pd.date_range(mon+'01', mon1+'01', freq=self.frequency)[:-1]

        # indices of the seletect month timesteps to apply to all variables
        # slices according data timesteps
        ts_slices = self.timesteps.slice_indexer(m_timesteps[0], m_timesteps[-1])
        # slices according monthly timesteps
        mts_slices = m_timesteps.slice_indexer(self.timesteps[0], self.timesteps[-1])
        print("MTS SLICES")
        print(mts_slices)


        nc_dims['time'] = m_timesteps.size
        cm_vars['time']['val'] = np.arange(nc_dims['time'])*float(self.frequency[:-1])
        cm_vars['time']['attrs']['units'] = cm_vars['time']['attrs']['units']\
                % m_timesteps[0].strftime('%Y-%m-%d %H:%M:%S')

        # Create a blank initilization array
        init_ndata = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        
        init_ndata[mts_slices, :] = cm_vars['ndata']['val'][ts_slices, :]
        cm_vars['ndata']['val'] = np.ma.masked_where(np.isnan(init_ndata), init_ndata)
        
        init_ncvar = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        # Update the time period with the data 
        init_ncvar[mts_slices, :] = nc_vars[ncvar]['val'][ts_slices, :]
        nc_vars[ncvar]['val'] = np.ma.masked_where(np.isnan(init_ncvar), init_ncvar)

        # output file namefter consulting with @sbasart and discussing with @hpetetin we came to the conclusion that given that MERRA-2 has 1-hourly fields at surf
        fname = os.path.join(outdir, "%s_%s.nc" % (ncvar, mon))

        # manage if dataset already exists
        if not os.path.exists(fname):
            # only if a new dataset is created we need to set dimensions
            # and variables
            fout = nc(fname, 'w')

            # Create netcdf dimensions
            for dim in nc_dims:
                print("Creating dimension '%s' with size '%s' ..." % (dim, nc_dims[dim]))
                fout.createDimension(dim, nc_dims[dim])

            # Create netcdf coordinates
            for varname in cm_vars:
                print("Creating variable '%s' with dims '%s' ..." % (varname,
                                                                     cm_vars[varname]['dims']))
                # Create the station_name variable
                if cm_vars[varname]['dtype'] == 'c':
                    print("****", varname, cm_vars[varname]['val'].shape, "****")
                    var = fout.createVariable(varname, cm_vars[varname]['dtype'],
                                              cm_vars[varname]['dims'], zlib=True)

                    var.setncatts(cm_vars[varname]['attrs'])
    
                    # Convert station name string to character variable
                    var[:] = \
                        nc4.stringtochar(np.array([v.decode('utf-8')
                                                   for v in cm_vars[varname]['val']]).\
                                                   astype('S%s' % str(nc_dims['strlen'])))
                # Create lon/lat/time vars
                elif varname in self._st_vars():
                    print("----", varname, cm_vars[varname]['val'].shape, "----")
                    var = fout.createVariable(varname, cm_vars[varname]['dtype'],
                                              cm_vars[varname]['dims'], zlib=True)
                    var.setncatts(cm_vars[varname]['attrs'])
                    var[:] = cm_vars[varname]['val']

                # Create ndata var
                else:
                    print("++++", varname, cm_vars[varname]['val'].shape,
                          cm_vars[varname]['dims'], "++++")
                    var = fout.createVariable(varname, cm_vars[varname]['dtype'],
                                              cm_vars[varname]['dims'], zlib=True,
                                              fill_value=self.fillval)
                    var.setncatts(cm_vars[varname]['attrs'])
                    tmp = cm_vars[varname]['val']
                    print("Masking the null value")
                    tmp = np.ma.masked_where(tmp == self.nullvalue, tmp)
                    var[:] = tmp
                    print("::::", var[:].shape, "::::")

            # Create the independent measured variable
            var = fout.createVariable(ncvar, nc_vars[ncvar]['dtype'], nc_vars[ncvar]['dims'],
                                      zlib=True, fill_value=self.fillval)
            var.setncatts(nc_vars[ncvar]['attrs'])
            tmp = nc_vars[ncvar]['val']
            tmp = np.ma.masked_where(tmp == self.nullvalue, tmp)
            print("Setting variable %s (shape %s) with shape %s from tmp with shape %s" % \
                (ncvar, var.shape, tmp[~tmp.mask].shape, tmp.shape))
            var[:] = tmp #nc_vars[ncvar]['val']

            
            # Close the dataset
            fout.close()
        else:
            # if the dataset already exists, we overwrite the data already there
            # variable
            tmp = nc_vars[ncvar]['val']
            
            if tmp[~tmp.mask].shape[0] != 0:
                fout = nc(fname, 'r+')
                fout.set_auto_mask(True)
                if ncvar in fout.variables:
                    # Copy variables already there
                    var = fout.variables[ncvar][:].copy()
                    var = np.ma.masked_where(np.isnan(var), var)

                    
                    # We need to overwrite the daily data
                    
                    print("Updating variable %s (shape %s) with shape %s from tmp with shape %s" % \
                        (ncvar, var.shape, tmp[~tmp.mask].shape, tmp.shape))
                    print("BEFORE", var[:][~tmp.mask][:10])
                    # I think it is here we need to overwrite everything 
                    var.mask[~tmp.mask] = False

                    # Update the points with the new data
                    var[~tmp.mask] = tmp[~tmp.mask]

                    var[:] = np.ma.masked_where(var == self.nullvalue, var)
                    


                    fout.variables[ncvar][:] = var.copy()
                    print("AFTER", var[:][~tmp.mask][:10])
                    
                    

                    # Also update the ndata variable and maybe the station_name variable
                    for varname in cm_vars:
                        print("Appending variable '%s' with dims '%s' ..." % (varname,
                                                                            cm_vars[varname]['dims']))
                        # Update ndata var variable
                        if varname == "ndata":
                            var = fout.variables[varname][:].copy()
                            var = np.ma.masked_where(np.isnan(var), var)
                            
                            tmp = cm_vars["ndata"]['val']
                    
                            print("Updating variable %s (shape %s) with shape %s from tmp with shape %s" % \
                                (varname, var.shape, tmp[~tmp.mask].shape, tmp.shape))
                            print("BEFORE", var[:][~tmp.mask][:10])

                            var.mask[~tmp.mask] = False
                            

                            # Update the points with the new data
                            var[~tmp.mask] = tmp[~tmp.mask]
                            var[:] = np.ma.masked_where(var == self.nullvalue, var)
                            


                            fout.variables[varname][:] = var.copy()
                            print("AFTER", var[:][~tmp.mask][:10])
                        # Station name
                        elif cm_vars[varname]['dtype'] == 'c':
                            print("****", varname, cm_vars[varname]['val'].shape, "****")
                            var = fout.variables[varname][:].copy()
                            
                            # Convert station name string to character variable
                            var[:] = nc4.stringtochar(np.array([v.decode('utf-8')
                                                        for v in cm_vars[varname]['val']]).\
                                                        astype('S%s' % str(nc_dims['strlen'])))
                            fout.variables[varname][:] = var.copy()
                            
                        # Create lon/lat/time vars
                        elif varname in self._st_vars():
                            print("----", varname, cm_vars[varname]['val'].shape, "----")
                            var = fout.variables[varname][:].copy()
                            
                            var[:] = cm_vars[varname]['val']
                            fout.variables[varname][:] = var.copy()

                            

                else:
                    # Create variable if it already exists
                    var = fout.createVariable(ncvar, nc_vars[ncvar]['dtype'], nc_vars[ncvar]['dims'],
                                              zlib=True, fill_value=self.fillval)
                    var.setncatts(nc_vars[ncvar]['attrs'])
                    tmp = nc_vars[ncvar]['val']
                    print("Setting variable %s (shape %s) with shape %s from tmp with shape %s" % \
                        (ncvar, var.shape, tmp[~tmp.mask].shape, tmp.shape))
                    var[:] = tmp #nc_vars[ncvar]['val']
                fout.close()
        os.system('chmod 766 -R {}'.format(fname))



    def write_netcdf(self, date=None):
        """ 
        Write NetCDF4 files according to department standards plus CF-1.6 
        """

        date = date or self.date
        # FIXME print one file per period specified (hardcoded: monthly)
        if (len(date) == 1) or (date[0][:6] == date[-1][:6]):
            # dlist = [datetime.strptime(date[0], "%Y%m%d"),]
            mfiles = [date[0][:6],]
        else:
            mfiles = list(set([dat[:6] for dat in date]))
            # dlist = pd.date_range(date[0], date[-1], freq='M')
        # mfiles = sorted(list(set([m.strftime("%Y%m") for m in dlist])))
        print("date", date, "MFILES", mfiles)
        mon = date[:6]
        # print one file per var
        for ncvar in self._nc_vars():
            # Calculate ndata value
            outdir = os.path.join(self.out_dir, ncvar)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            print(mon, ncvar, outdir)
            self.write_monthly_var(mon, ncvar, outdir)


    def retrieve_stations(self, st_file=None):
        """ 
        Read txt file and load coords, to be called after the AERONET 
        """

        inp_file = st_file or self.st_file
        print("Station file:", inp_file)

        all_stations = pd.read_csv(inp_file, skiprows=1, header=0, index_col=0)

        lon = all_stations[self.st_lons].values
        lat = all_stations[self.st_lats].values
        alt = all_stations[self.st_alts].values
        self.station_name = all_stations.index.values

        self.station_name_full = np.array(["NaN" for i in range(0, self.station)], dtype="|S80") #dytpe specified so that full station name is displayed
        self.lon = np.ma.masked_array(np.ones((5000))*np.nan)
        self.lat  = np.ma.masked_array(np.ones((5000))*np.nan)
        self.alt = np.ma.masked_array(np.ones((5000))*np.nan)

        self.station_name_full[:self.station_name.shape[0]] = self.station_name
        self.lon[:lon.shape[0]] = lon
        self.lat[:lat.shape[0]] = lat
        self.alt[:alt.shape[0]] = alt
        
        self.station_count = np.array([i + 1 for i in range(0, self.station)])
        


    def get_dataframe(self, infile):
        """ 
        Use Pandas to read csvile and move to dataframes
        """
        
    
        print('File', infile)
        with open(infile) as htmlfile:
            html_string = ''.join(htmlfile.readlines()[7:-1]).replace('<br>', '')
            csvdata = StringIO(html_string)

        tdf = pd.read_csv(csvdata, engine='python', header=0,
                              usecols=self.columns,
                              index_col=False,
                              dtype = {'col1': str}
                              )

        tdf["datetime"] = pd.to_datetime(
            tdf[self.dt_cols[0]].astype(str) + ' ' + tdf[self.dt_cols[1]].astype(str), 
            format='%d:%m:%Y %H:%M:%S'
            )
        tdf = tdf.set_index("datetime")

        
        mval = -999.
        # replace -999. with np.nan for columns associated with the final output wavelength
        # Making it compatible with directsun and oneill
        for column in self.wavelength_columns:
            tdf[column] = tdf[column].astype(float)
            tdf[column] = np.where(tdf[column] == mval, np.nan, tdf[column])

        return tdf


    def retrieve_aeronet_data(self, date=None):
        """ Read html/csv files daily downloaded from AERONET website """

        date = date or self.date
        infile = self.inp_tpl % date # (date)
        
        dataframe = self.get_dataframe(infile)
        
        # group by station
        self.gdataframe = dataframe.groupby(self.columns[0])
        
        self.retrieve_stations(st_file=self.st_file)
        #for year in years:
        print("AFTER", self.station)

        dims = (self.t_steps, self.station)

        # Create this as a dictionary instead
        self.analysis_variables_dict = {}
        for k in self.analysis_variables:
            self.analysis_variables_dict[k] = np.empty(dims)
            self.analysis_variables_dict[k][:] = np.nan


    def save_original_files_if_wrong_stations(self, date=None):
        """
        A function which saves original files if a wrong station appears in the data
        """
        date = date or self.date
        
        infile = self.inp_tpl % date # (date)
        
            
        # Read in file again 
        print('File', infile)
        with open(infile) as htmlfile:
            html_string = ''.join(htmlfile.readlines()[7:-1]).replace('<br>', '')
            csvdata = StringIO(html_string)

        tdf = pd.read_csv(csvdata, engine='python', header=0,
                            usecols=self.columns,
                            index_col=False,
                            dtype = {'col1': str},)
        tdf["datetime"] = pd.to_datetime(
            tdf[self.dt_cols[0]].astype(str) + ' ' + tdf[self.dt_cols[1]].astype(str), 
            format='%d:%m:%Y %H:%M:%S'
            )
        
        tdf = tdf.set_index("datetime")
        
        # check if any of the odd stations are in the original files
        
        stations_in_df = (tdf["AERONET_Site"].isin(STATIONS)).any()

        if stations_in_df:
            print("ALERT: Odd stations in the original files")
            # Output the data to an odd stations folder in the form {date}_{current_date}.csv
            current_date = datetime.today().strftime("%Y%m%d")
            print(current_date)
            outfile = "{}/{}_{}.csv".format("/".join(infile.split("/")[:-1]), date, current_date)
            print(outfile)
            tdf.to_csv(outfile, index=False)
             


#### Direct Sun Formatting Lev1.5
class AERONETDirectSun(FormatAERONETv3):
    def __init__(self, *args):
        FormatAERONETv3.__init__(self, *args)
        self.inp_folder = '/shared/data/obs/nonghost/nasa-aeronet/directsun_v3-lev{}/original_files'.format(self.level) # ADD SHARED PATH HERE
        self.inp_tpl = self.inp_folder + '/%s.html'
        # /esarchive/obs/nasa-aeronet/directsun_v3-lev15/original_files/%s.html
        # '/esarchive/scratch/cmeikle/Projects/data/nasa-aeronet/original_files/directsun_v3-lev15/%s.html'
        # /esarchive/scratch/cmeikle/Projects/data/nasa-aeronet/directsun_v3-lev{}/3hourly
        # /esarchive/obs/nasa-aeronet/directsun_v3-lev{}/3hourly_new
        if self.frequency == '3H':
            self.out_dir = '/shared/trainees/tvintimi/obs/nonghost/nasa-aeronet/directsun_v3-lev{}/3hourly'.format(self.level) # ADD PATH HERE
        elif self.frequency == '1H':
            self.out_dir = '/shared/trainees/tvintimi/obs/nonghost/nasa-aeronet/directsun_v3-lev{}/hourly'.format(self.level) # ADD PATH HERE

        self.columns = ["AERONET_Site", "Date(dd:mm:yyyy)", "Time(hh:mm:ss)",
                        "AOD_440nm", "AOD_675nm", "AOD_870nm",
                        "440-870_Angstrom_Exponent"]
        self.dt_cols = ["Date(dd:mm:yyyy)", "Time(hh:mm:ss)"]
        self.wavelength_columns = ["AOD_440nm", "AOD_675nm", "AOD_870nm"]
        self.aod_var = 'AOD_550nm'
        self.arm_var = '440-870_Angstrom_Exponent'
        self.analysis_variables = ['ae440-870aero', 'od550aero', 'ndata']
        
    def _nc_vars(self):
        """
        Direct Sun data variables
        """
        return {
            'ae440-870aero': {
                'dims': ('time', 'station'),
                'dtype': 'f',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "Total aerosol Angstrom parameter [470-870]", 
                },
                'val': self.ae440_870aero,
            },
            'od550aero': {
                'dims': ('time', 'station'),
                'dtype': 'f',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "aerosol optical depth at 550nm",
                },
                'val': self.od550aero,
            },
        }



    def _var_calc(self, dataframe):
        """ AOD 550nm calculation """
        # validity filters
        # print(dataframe.columns)
        dataframe.loc[(dataframe['AOD_440nm'] < 0.) &
                      (dataframe['AOD_675nm'] < 0.) &
                      (dataframe['AOD_870nm'] < 0.) &
                      (dataframe[self.arm_var] < 0.)] = np.nan
        # drop row with nan or with aod < -0.001
        # dataframe = dataframe[dataframe>=-0.001].dropna()

        # retrieve AODs and AE
        aod_440 = dataframe['AOD_440nm']
        aod_675 = dataframe['AOD_675nm']
        aod_870 = dataframe['AOD_870nm']
        arm_440_870 = dataframe[self.arm_var]

        # calc AOD550nm
        dataframe[self.aod_var] = (1.0/3) * (
            aod_440*((440.0/550)**arm_440_870) +
            aod_675*((675.0/550)**arm_440_870) +
            aod_870*((870.0/550)**arm_440_870)
            )
        dataframe = dataframe[[self.aod_var, self.arm_var]].\
                resample(self.frequency.lower()).apply(['mean', 'count'])
                         #base=self.base,
                         #loffset=self.loffset).apply(['mean', 'count']
                         
        

        return dataframe.reindex(self.timesteps)



    def retrieve_aeronet_data(self, date=None):
        """
        Child of the main aeronet class for just dealing with direct sun data
        """
        FormatAERONETv3.retrieve_aeronet_data(self, date=date)
        # create and initialize arrays (8 timesteps x X stations)
        nc_dims = self._nc_dims()
        
        od550aero = self.analysis_variables_dict['od550aero']
        ae440_870aero = self.analysis_variables_dict['ae440-870aero']
        ndata = self.analysis_variables_dict['ndata']

        

        self.newg = None
        print(self.gdataframe.count())
        
        for st_name in self.station_name:
            
            try:
                oldg = self.gdataframe.get_group(st_name)
            except: 
                #print("There is no station name group in the dataframe")
                continue
            
            self.station_idx = self.station_name.tolist().index(st_name)
            # calculate data
            
            self.newg = self._var_calc(oldg[oldg.columns[1:]].copy())
            od550aero[:, self.station_idx] = self.newg[self.aod_var]['mean'].values
            ae440_870aero[:, self.station_idx] = self.newg[self.arm_var]['mean'].values
            # Set new counts to ndata values
            ndata[:, self.station_idx] = self.newg["AOD_550nm"]['count'].values
            


        # If the ncvars are nan, add a missing value 
        # so that all values for that day are updated in the netcdf
        # This is then changed back to nan in the write monthly var function
        od550aero = np.where(od550aero >= self.nullvalue, od550aero, self.nullvalue)
        ae440_870aero = np.where(ae440_870aero >= self.nullvalue, ae440_870aero, self.nullvalue)
        ndata = np.where(ndata >= self.nullvalue, ndata, self.nullvalue)


        # set variables
        print("Setting variables ...")
        self.time = self.newg.index.hour
        nc_dims['time'] = self.time.size
        # Create empty masked arrays
        self.od550aero = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        self.ae440_870aero  = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        self.ndata = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        

        # Append data to the arrays
        od550aero = np.ma.masked_where(np.isnan(od550aero), od550aero)
        ae440_870aero = np.ma.masked_where(np.isnan(ae440_870aero), ae440_870aero)
        ndata = np.ma.masked_where(np.isnan(ndata), ndata)
        
        self.od550aero[:od550aero.shape[0], :od550aero.shape[1]] = od550aero
        self.ae440_870aero[:ae440_870aero.shape[0], :ae440_870aero.shape[1]] = ae440_870aero
        self.ndata[:ndata.shape[0], :ndata.shape[1]] = ndata
        print("NDATA")
        print(ndata)
        print(ndata.shape)

        

        self.write_netcdf()




#### Oneill Formatting Lev1.5
class AERONETOneill(FormatAERONETv3):
    def __init__(self, *args):
        FormatAERONETv3.__init__(self, *args)
        self.inp_folder = '/shared/data/obs/nonghost/nasa-aeronet/oneill_v3-lev{}/original_files'.format(self.level) # ADD SHARED PATH HERE
        self.inp_tpl = self.inp_folder + '/%s.html'
        # /esarchive/obs/nasa-aeronet/oneill_v3-lev15/original_files/%s.html
        # '/esarchive/scratch/cmeikle/Projects/data/nasa-aeronet/original_files/oneill_v3-lev15/%s.html'
        # /esarchive/scratch/cmeikle/Projects/data/nasa-aeronet/oneill_v3-lev{}/3hourly
        # /esarchive/obs/nasa-aeronet/oneill_v3-lev{}/3hourly_new
        if self.frequency == '3H':
            self.out_dir = '/shared/trainees/tvintimi/obs/nonghost/nasa-aeronet/oneill_v3-lev{}/3hourly'.format(self.level) # ADD PATH HERE
        elif self.frequency == '1H':
            self.out_dir = '/shared/trainees/tvintimi/obs/nonghost/nasa-aeronet/oneill_v3-lev{}/hourly'.format(self.level) # ADD PATH HERE
        
        
        self.columns = ['AERONET_Site', 'Date_(dd:mm:yyyy)', 'Time_(hh:mm:ss)', 
                        'Total_AOD_500nm[tau_a]','Fine_Mode_AOD_500nm[tau_f]',
                        'Coarse_Mode_AOD_500nm[tau_c]']
        self.dt_cols = ["Date_(dd:mm:yyyy)", "Time_(hh:mm:ss)"]
        self.wavelength_columns = ['Total_AOD_500nm[tau_a]','Fine_Mode_AOD_500nm[tau_f]','Coarse_Mode_AOD_500nm[tau_c]']
        self.od_500 = 'Total_AOD_500nm[tau_a]'
        self.od_500_fine = 'Fine_Mode_AOD_500nm[tau_f]'
        self.od_500_coarse = 'Coarse_Mode_AOD_500nm[tau_c]'
        self.analysis_variables = ['od500aero', 'od500aerofine', 'od500aerocoarse', 'ndata']

        
    def _nc_vars(self):
        """
        Oniell data variables
        """
        return {
            'od500aero': {
                'dims': ('time', 'station'),
                'dtype': 'f',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "aerosol optical depth at 500nm",
                },
                'val': self.od500aero,
            },
            'od500aerofine': {
                'dims': ('time', 'station'),
                'dtype': 'f',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "aerosol optical depth at 500nm for fine fractions reff<0.6",
                },
                'val': self.od500aerofine,
            },
            'od500aerocoarse': {
                'dims': ('time', 'station'),
                'dtype': 'f',
                'attrs': {
                    'missing_value': self.fillval,
                    'units': '-',
                    'long_name': "aerosol optical depth at 500nm for coarse fractions reff>0.6",
                },
                'val': self.od500aerocoarse,
            },
        }
    
    def _var_calc(self, dataframe):
        """
        Resample the variables into hourly or 3 hourly periods
        """
        
        dataframe = dataframe[[self.od_500, self.od_500_fine, self.od_500_coarse]].\
                resample(self.frequency.lower()).apply(['mean', 'count'])
                         #base=self.base,
                         #loffset=self.loffset).apply(['mean', 'count'])

        return dataframe.reindex(self.timesteps)


    def retrieve_aeronet_data(self, date=None):
        FormatAERONETv3.retrieve_aeronet_data(self, date=date)
        nc_dims = self._nc_dims()
        od500aero = self.analysis_variables_dict['od500aero']
        od500aerofine = self.analysis_variables_dict['od500aerofine']
        od500aerocoarse = self.analysis_variables_dict['od500aerocoarse']
        ndata = self.analysis_variables_dict['ndata']


        #Need to rewrite this I think to get it in the right format
        self.newg = None
        for st_name in self.station_name:
            try:
                oldg = self.gdataframe.get_group(st_name)
            except: continue
            #print('Station', st_name)
            self.station_idx = self.station_name.tolist().index(st_name)
            self.newg = self._var_calc(oldg[oldg.columns[1:]].copy())
            od500aero[:, self.station_idx] = self.newg[self.od_500]['mean'].values
            od500aerofine[:, self.station_idx] = self.newg[self.od_500_fine]['mean'].values
            od500aerocoarse[:, self.station_idx] = self.newg[self.od_500_coarse]['mean'].values
            
            # Set new counts to ndata values
            ndata[:, self.station_idx] = self.newg["Total_AOD_500nm[tau_a]"]['count'].values

            

        # If the ncvars are nan, add a missing value 
        #so that all values for that day are updated in the netcdf
        # This is then changed back to nan in the write monthly var function
        

        od500aero = np.where(od500aero >= self.nullvalue, od500aero, self.nullvalue)
        od500aerofine = np.where(od500aerofine >= self.nullvalue, od500aerofine, self.nullvalue)
        od500aerocoarse = np.where(od500aerocoarse >= self.nullvalue, od500aerocoarse, self.nullvalue)
        ndata = np.where(ndata >= self.nullvalue, ndata, self.nullvalue)




        print("Setting variables ...")
        self.time = self.newg.index.hour
        nc_dims['time'] = self.time.size
        self.od500aero = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        self.od500aerofine  = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        self.od500aerocoarse  = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)
        self.ndata = np.ma.masked_array(np.ones((nc_dims['time'], nc_dims['station']))*np.nan)


        od500aero = np.ma.masked_where(np.isnan(od500aero), od500aero)
        od500aerofine = np.ma.masked_where(np.isnan(od500aerofine), od500aerofine)
        od500aerocoarse = np.ma.masked_where(np.isnan(od500aerocoarse), od500aerocoarse)
        ndata = np.ma.masked_where(np.isnan(ndata), ndata)


        self.od500aero[:od500aero.shape[0], :od500aero.shape[1]] = od500aero
        self.od500aerofine[:od500aerofine.shape[0], :od500aerofine.shape[1]] = od500aerofine
        self.od500aerocoarse[:od500aerocoarse.shape[0], :od500aerocoarse.shape[1]] = od500aerocoarse 
        self.ndata[:ndata.shape[0], :ndata.shape[1]] = ndata
        print("NDATA")
        print(ndata)
        print(ndata.shape)

        self.write_netcdf()


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-s", "--startdate", help="Start date of the formatting.")
    parser.add_argument("-e", "--enddate", help="End date of the formatting.")
    parser.add_argument("-f", "--frequency", help="Frequency of the timesteps 3H or 1H.")
    parser.add_argument("-v", "--aeronet_variable", help="Which Aeronet variable do we need to format (Directsun/Oniell).") # 'ds' for directsun 'on' for 
    parser.add_argument("-l", "--level", help="Level to be passed, either 1.5 or 2 or 1.5_validated")
    options = parser.parse_args(args)
    return options




if __name__ == "__main__":
    print(sys.argv)

    options = getOptions(sys.argv[1:])
    start_date = options.startdate
    end_date = options.enddate
    frequency = options.frequency
    aeronet_variable = options.aeronet_variable
    level = options.level

    # Set defaults
    if end_date == None:
        end_date = datetime.today().strftime('%Y%m%d')
        #end_date == datetime(2025,3,9)
    if start_date == None:
        start_date = end_date
        #start_date == datetime(2025,3,1) 
    if frequency == None:
        frequency = '3H'
    if aeronet_variable == None:
        aeronet_variable = "ds"
    if level == None:
        level = "15"

    print("Running formatting for follwing variables: Startdate - {}, Enddate - {}, Frequency - {}, Aeronet Variable - {}, level - {}".format(
        start_date, end_date, frequency, aeronet_variable, level
        ))
    
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start_date, end_date)]

    

    for date in dates:
        # Update this to include level
        if aeronet_variable == "ds": # Run direct sun formatting
            cl = AERONETDirectSun(date, frequency, level)
        elif aeronet_variable == "on": # Run Oniell formatting
            cl = AERONETOneill(date, frequency, level)
        else:
            print("Aeronet variable not recognised")
    
        # This has now initialized everything
        print("Formatting for date %s ..." % cl.date)
        cl.retrieve_aeronet_data()
        print("Checking original files for date %s ..." %cl.date)
        cl.save_original_files_if_wrong_stations()

