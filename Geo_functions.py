# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:35:21 2021

@author: thoma
"""
import gdal
import rasterio as r
from rasterio.mask import mask
import os
import fiona
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape 
import geopandas as gpd
from rasterstats import zonal_stats
import netCDF4
import osr
from statsmodels.tsa.seasonal import seasonal_decompose
from osgeo import ogr
import progressbar
from time import sleep

###### Create a geotiff masked by a shapefile. ################## 
# Input : - shape_file = the mask as a shapefile   (str)        #
#         - tif_file = the original geotiff file    (str)       #
# Output  - output_file = the masked geotiff you want to create # 
#################################################################

def shape_mask_tif(shape_file,tif_file,output_tif):
    with fiona.open(shape_file) as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]
        
    # load the raster, mask it by the polygon and crop it
    with r.open(tif_file) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
    out_meta = src.meta.copy()
    
    # save the resulting raster  
    out_meta.update({"driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
    "transform": out_transform})
    
    with r.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)

###### Create a shapefile with buffer around points of a shapefile. ###################### 
# Input : - spa = Dataframe with Point coordonates as columns (Longitude,Latitude) (df)  #
#         - buffer = length of the buffer radius IN DEGREES    (float)                   #
# Output  - output_shp = the shapefile you want to create                                # 
##########################################################################################

def create_shape_buffer(spa,buffer,output_shp):
    spa.index=spa.name
    gs_spa=[]
    for i in range(len(spa)):
        gs_spa.append(Point(spa.Longitude[i],spa.Latitude[i]))
    gs_buffer=gpd.GeoSeries(gs_spa).buffer(float(buffer))
    gs_buffer.to_file(output_shp, driver='GeoJSON')    

####################### Extract TS from folder of geotiff ##############################
# Input :  - folder = Folder with the geotiff files (str)                              #
#          - shapefile = Extract info from this shapefile  (shp)                       #
#          - ind_all = Pandas daterange of the geotiff files (DatetimeIndex)           #
#          - resample = Resampling frequency (values='W','M','D' or 'None')            #
#          - ext_m = Extraction method ('mean','min','max',etc...)                     #
# Output : - TS extracted  (Series)                                                    #
########################################################################################

def extract_TS_from_geo(folder,shapefile,ind_all,resample='None',ext_m='mean'):
    hum_tot=pd.DataFrame()
    for filename in os.listdir(folder):
        test=zonal_stats(shapefile, folder+filename,
        stats=ext_m)
        hum=[]
        for i in range(len(test)):
            if test[i]['mean'] != float('-inf'):
                hum.append(test[i]['mean'])
            else:
                hum.append(float('NaN'))
        hum=np.array(hum).reshape(1,len(hum))
        hum_tot=pd.concat([hum_tot,pd.DataFrame(hum)])  
    hum_tot.index=ind_all
    hum_tot=hum_tot.astype(float)
    if resample != 'None':
        hum_tot=hum_tot.resample(resample).mean()
    
    return hum_tot

####################### Extract folder of geotiff from nc4 #############################
# Input :  - nc4 = Nc4 file (nc4)                                                      #
#          - layer = Nc4 file's layer to extract    (str)                              #
#          - ind_all = Pandas daterange of the nc4 file (DatetimeIndex)                #
# Output : - folder = Folder with the geotiff files (str)                              #
########################################################################################

def extract_geo_from_nc4(nc4,layer,ind_all,folder):

    ncfile = netCDF4.Dataset(nc4)
    long = ncfile.variables['longitude'][:]
    lat = ncfile.variables["latitude"][:]
    rr = ncfile.variables[layer][:]
    
    for i in range(len(rr)):
        dataw=  rr[i]
        nx = len(long)
        ny = len(lat)
        xmin, ymin, xmax, ymax = [long.min(), lat.min(), long.max(), lat.max()]
        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        dst_ds = gdal.GetDriverByName('GTiff').Create(folder+str(ind_all[i][0])+'.tif', nx, ny, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)        # specify coords
        srs = osr.SpatialReference()                # establish encoding
        srs.ImportFromEPSG(4326)                    # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())     # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(dataw)   # write r-band to the raster
        dst_ds.FlushCache()                         # write to disk
        dst_ds = None  



####################### Extract TS from folder of geotiff ############################################
# Function to extract global country time series of precipitation, temperature and humidity from     #
# folder of geotiff. Every pixel (10x10 km) with crops weight the mean Time series depending on the  #
# season and the surface of exploitation.                                                            #
#                                                                                                    #
# Input :  - shape_country = shapefile of the country (shp)                                          #
#          - n_country = name of the country (str)                                                   #
#          - ind_all = Pandas daterange of the geotiff files fold_v (DatetimeIndex)                  #
#          - fold_ext = folder where the buffers will be created                                     #
#          - df_cr = Mirca 2000 dataframe                                                            #
#          - fold_v = Folder with the variables geotiff files (Temperature, Precipitation, Humidity) #
#          - fold_output = folder where the created .csv will be                                     #
# Output : - DataFrame as .csv of Temperature, Preciptation and Humidity residuals  (csv)            #
######################################################################################################

def extract_TS_from_crop_country(shape_country,n_country,ind_all,
                                 fold_ext='/media/thomas/OS/Users/ThomasSchinca/Pace_data/shape_ext/',
                                 df_cr='/media/thomas/OS/Users/ThomasSchinca/Pace_data/CELL_SPECIFIC_CROPPING_CALENDARS.txt',
                                 fold_v='/media/thomas/OS/Users/ThomasSchinca/Pace_data/Images/Total/',
                                 fold_output='/media/thomas/OS/Users/ThomasSchinca/Pace_data/Classif/',
                                 fold_data = '/media/thomas/OS/Users/ThomasSchinca/Pace_data/Data/'):
                                
####### Name of the country file creation 
    
    n_country_link=n_country.replace(" ", "_")    
    
    if n_country in os.listdir(fold_ext):
        shape_fold=fold_ext+n_country_link+'/'
    else : 
        os.mkdir(fold_ext+n_country_link+'/')
        shape_fold=fold_ext+n_country_link+'/'
    
    # lat long extrem borders with shape file 
    input_shp = ogr.Open(shape_country)
    shp_layer = input_shp.GetLayer()
    minx, maxx, miny, maxy = shp_layer.GetExtent()    # x = long, y = lat 
    
    ###### Crop information extraction 
    
    df=pd.read_csv(df_cr,delim_whitespace=True)    
    df=df[(df.lat>miny) & (df.lat<maxy) & (df.long>minx) & (df.long<maxx)]
    multipol = fiona.open(shape_country)
    multi= next(iter(multipol)) # only one feature in the shapefile
    df_c=[]
    bar = progressbar.ProgressBar(maxval=len(df.long.unique())*len(df.lat.unique()), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i=0
    for lon in df.long.unique():
        for late in df.lat.unique():
            if Point(lon,late).within(shape(multi['geometry'])):
                df_c.append(df[(df.long==lon) & (df.lat==late)])
                bar.update(i+1)
                sleep(0.1)
            i=i+1
    df_c=pd.concat(df_c)        
    df_c=df_c.reset_index(drop=True)          
    print('\n')
    print('Crop pixels extracted')
    print('\n')
    crop_p=pd.DataFrame(columns=range(1,13))
    for i in range(len(df_c)):
        l=[0]*12
        if df_c.start[i]>df_c.end[i]:
            l[0:df_c.end[i]]=[1]*len(l[0:df_c.end[i]])
            l[df_c.start[i]-1:]=[1]*len(l[df_c.start[i]-1:])
        else :
            l[(df_c.start[i]-1):df_c.end[i]]=[1]*len(l[(df_c.start[i]-1):df_c.end[i]])
        crop_p.loc[i,:]=l    
    df_c=df_c.drop(['start','end'],axis=1)
    df_c=pd.concat([df_c,crop_p],axis=1)

    ######## Creation of the pixels shapefile 
    
    for i in range(len(df_c[['lat','long']].drop_duplicates())):
        gs_spa=[]
        gs_spa.append(Point(df_c[['lat','long']].drop_duplicates()['long'].iloc[i],df_c[['lat','long']].drop_duplicates()['lat'].iloc[i]))
        gs_buffer=gpd.GeoSeries(gs_spa).buffer(float(0.11784641/2))
        gs_buffer.to_file(shape_fold+str(df_c.Cell_ID.unique()[i])+'.shp')        

    ########## Extraction of the mean value of the variables of each pixel 
    
    df_clim_tot=[]
    for variable in ['Temp','RR','Hum']:
        bar = progressbar.ProgressBar(maxval=len(df_c.Cell_ID.unique()), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        cont=0
        df_clim=[]
        for elmt in df_c.Cell_ID.unique(): 
            df_tot=[]
            for filename in os.listdir(fold_v+variable+'/'):
                test=zonal_stats(shape_fold+str(elmt)+'.shp', fold_v+variable+'/'+filename,
                stats="mean", nodata=-32767)
                for i in range(len(test)):
                    if test[i]['mean'] != float('-inf'):
                        hum=test[i]['mean']
                    else:
                        hum=float('NaN')
                df_tot.append(hum) 
            df_clim.append(df_tot)
            bar.update(cont+1)
            sleep(0.1)
            cont=cont+1
        print('\n')
        print(variable, ' done.')
        df_clim_2=pd.DataFrame(np.array(df_clim).T)
        df_clim_2.index=ind_all
        df_clim_2.columns=df_c.Cell_ID.unique()
        bar.finish() 
        df_clim_tot.append(df_clim_2)    
    
    ############ Interpolation with the nearest neighboor 
    # from right to left
    for i in range(len(df_clim_tot)): 
        df_clim_tot[i]=df_clim_tot[i].fillna(method='bfill',axis=1)
    # from left to right    
    for i in range(len(df_clim_tot)): 
        df_clim_tot[i]=df_clim_tot[i].fillna(method='ffill',axis=1)
            
    ############ Application of the crop weight on the Climatic TS
    
    weight=pd.DataFrame()    
    for elmt in df_c.Cell_ID.unique():    
        df_c_u=df_c[df_c.Cell_ID==elmt]
        l_m=[]
        for months in range(1,13):
            val=sum(df_c_u[df_c_u[months]==1]['area'])
            l_m.append(val)
        weight[str(elmt)]=l_m    
    weight.index=range(1,13)        
    
    ts_clim_final=pd.DataFrame()
    for variable in ['Temp','RR','Hum']:
        df_v = df_clim_tot[['Temp','RR','Hum'].index(variable)]
        val=[]
        for date in df_v.index: 
            val.append(sum(weight.loc[date.month].multiply(df_v.loc[date].values))/sum(weight.loc[date.month]))
        ts_clim_final[variable]=val 
    ts_clim_final.index=ind_all    
    
    ############### Normalization 
    
    n_ts_clim_final=(ts_clim_final-ts_clim_final.mean())/ts_clim_final.std()
    
    ############### Removing the seasonal component 
    
    f_df_clim= pd.DataFrame()
    for n_series in list(n_ts_clim_final.columns):
        result_add = seasonal_decompose(n_ts_clim_final[n_series], model='additive', extrapolate_trend='freq')
        diff = list()
        for i in range(len(result_add.seasonal)):
        	value = n_ts_clim_final[n_series][i] - result_add.seasonal[i]
        	diff.append(value)
        f_df_clim[n_series]=diff 
    f_df_clim.index=df_clim_2.index 

    ################ Create the output csv    

    os.mkdir(fold_data+n_country+'/')    
    f_df_clim.to_csv(fold_output+n_country+'.csv')
    n_ts_clim_final.to_csv(fold_data+n_country+'/Norm.csv')
    ts_clim_final.to_csv(fold_data+n_country+'/Raw.csv')
    return (ts_clim_final,n_ts_clim_final,f_df_clim)

