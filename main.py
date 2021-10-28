# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:17:26 2021

@author: thoma
"""
#### Packages 

import pandas as pd 
import numpy as np 
import os 
from shapelets_lts.classification import LtsShapeletClassifier


os.chdir('C:/Users/thoma/Desktop/PhD/Pace/Python')

#### Functions/Classes

from Geo_functions import *
from Process_data import *

########################### Creation/extraction of the climatic ts 
# Features
shape_file_fold='D:/Pace_data/shapes_c/'
ind_m=pd.date_range(start='01/01/1988',end='31/12/2020',freq='M')

# List of the shp file available
shape_cont=[]
for elmt in os.listdir(shape_file_fold): 
    shape_cont.append(elmt[:-4])
shape_cont=list(set(shape_cont))

# Create the missing ones
missing_c=detect_miss_c('C:/Users/thoma/Desktop/PhD/Pace/Python/Classif/', shape_cont)
for elmt in missing_c:    
    extract_TS_from_crop_country(shape_file_fold+elmt+'.shp',elmt,ind_m)  

########################### Preprocess the data

min_years=2
l_country=country_clim_check('C:/Users/thoma/Desktop/PhD/Pace/Python/Classif/')
df_onset= create_df_conf(pd.read_csv('D:/Pace_Data/ucdp-prio-acd-211.csv',index_col=0),'01/01/1988','31/12/2020',min_years,l_country)
l_c_s=fill_clim_seq(df_onset,'C:/Users/thoma/Desktop/PhD/Pace/Python/Classif/',min_years)



