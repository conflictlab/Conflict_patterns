# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:51:01 2021

@author: thoma
"""

import pandas as pd 
import os
import numpy as np 
import datetime as dt
import random
from tslearn.metrics import dtw
import statistics as st 

########################################################## Class ########################################################################################################################

###### Class to stock the climatic TS  ############################## 
# Attributes: - country   (str)                                     #
#             - ind_c = number of TS of this country    (ind)       #
#             - TS (Temp,RR,Hum) = Climatic time series (pd.Series) #
#             - dates (DateIndex)                                   #
#####################################################################

class Climate_seq():
    
    def __init__(self,country,ind_c,Temp,RR,Hum,dates):
        self.country = country
        self.ind_c = ind_c
        self.Temp = Temp
        self.RR = RR
        self.Hum = Hum
        self.dates = dates


######################################################### Function #######################################################################################################################


###### Create a list of the Climatic sequences (class) ######################## 
# Input : - df_onset = DataFrame of the selected onsets   (pd.DataFrame)      #
#             -> Output of create_df_conf function                            #
#         - meteo_fold = Folder with the csv file of meteo time series (str)  #
#         - min_years = Minimum number of years between two onsets (int)      #
# Output  - l_c_s = list of the Climatic sequences (list)                     # 
###############################################################################

def fill_clim_seq(df_onset,meteo_fold,min_years):
    l_c_s=[]
    df_onset=df_onset.loc[:,(df_onset!=0).any(axis=0)]
    for i in range(len(df_onset.iloc[0,:])): 
        cont=0
        meteo=pd.read_csv(meteo_fold+df_onset.columns[i]+'.csv',index_col=(0),parse_dates=True)
        for j in range(len(df_onset)):
            if df_onset.iloc[j,i]==1:
                ind_ext=pd.date_range(df_onset.index[j]-dt.timedelta(weeks=min_years*52), df_onset.index[j],freq='M')
                l_c_s.append(Climate_seq(df_onset.columns[i], cont, meteo.loc[ind_ext,'Temp'], meteo.loc[ind_ext,'RR'], meteo.loc[ind_ext,'Hum'], ind_ext))
                cont=cont+1
    return l_c_s

###### Check if the wanted countries are in the weather dataset  ############## 
# Input : - fold = Folder with the csv file of meteo time series (str)        #
#         - wanted_c = list of countries wanted (list of str)                 #
#               -> if 'None', all the countries of the fold are included      #
# Output  - l_country = list of the countries (list of str)                   # 
#             -> if country missing = error raised                            #
###############################################################################
        
def country_clim_check(fold,wanted_c=None):
    l_country_2=[]
    for elmt in os.listdir(fold): 
        l_country_2.append(elmt[:-4])
    if wanted_c==None:    
        l_country=list(set(l_country_2))  
    else :
        for elmt in wanted_c:
            if elmt not in l_country_2:
                raise NameError('This country is not in the folder')
        l_country = wanted_c        
    return l_country

###### Detect the wanted countries if not in the weather dataset  ############# 
# Input : - fold = Folder with the csv file of meteo time series (str)        #
#         - wanted_c = list of countries wanted (list of str)                 #
# Output: - mis_country= list of the missing country (list of str)            #
###############################################################################
        
def detect_miss_c(fold,wanted_c):
    l_country_2=[]
    for elmt in os.listdir(fold): 
        l_country_2.append(elmt[:-4])
    mis_country=[]    
    for elmt in wanted_c:
        if elmt not in l_country_2:
            mis_country.append(elmt)
    return mis_country

###### Create a list of the Climatic sequences (class) ############################# 
# Input : - df_ons = DataFrame of the onsets   (pd.DataFrame)                      #
#             -> UCDP/PRIO Armed Conflict Dataset version 21.1                     #
#         - start/end = start and end of the study period  ('%DD/%MM/%YYYY')       # 
#         - min_years = Minimum number of years between two onsets (int)           #
#         - l_country = Output of the country_clim_check function                  #
# Output  - df_onset = df_onset = DataFrame of the selected onsets (pd.DataFrame)  # 
####################################################################################

def create_df_conf (df_ons,start,end,min_years,l_country):
    
    df_ons=df_ons[(df_ons.type_of_conflict==3) | (df_ons.type_of_conflict==4)]
    df_ons=df_ons[df_ons['location'].isin(l_country)]
    df_ons=df_ons[df_ons.start_date>start[6:]+'-'+start[3:5]+'-'+start[0:2]]
    df_onset=pd.DataFrame(index=pd.date_range(start=start,end=end,freq='M'))
    for name in l_country:
        df_ons_sub=df_ons[df_ons.location==name]
        date_l=df_ons_sub.start_date.unique()
        date_l=pd.DataFrame(date_l).sort_values(by=0)
        
        df_ons_sub=pd.DataFrame(data=[0]*len(pd.date_range(start=start,end=end,freq='D')),index=pd.date_range(start=start,end=end,freq='D'))
        
        if len(date_l)>0:
            df_ons_sub.loc[dt.datetime(int(date_l.iloc[0,0][0:4]),int(date_l.iloc[0,0][5:7]), int(date_l.iloc[0,0][8:10]))]=1
            for i in range(len(date_l.loc[1:,:])): 
                if (dt.datetime(int(date_l.iloc[(i+1),0][0:4]),int(date_l.iloc[(i+1),0][5:7]), int(date_l.iloc[(i+1),0][8:10]))-dt.datetime(int(date_l.iloc[i,0][0:4]),int(date_l.iloc[i,0][5:7]), int(date_l.iloc[i,0][8:10]))).days > min_years*366 : 
                    df_ons_sub.loc[dt.datetime(int(date_l.iloc[i,0][0:4]),int(date_l.iloc[i,0][5:7]), int(date_l.iloc[i,0][8:10]))]=1
            
        df_ons_sub=df_ons_sub.resample('M').max()        
        df_onset[name]=df_ons_sub
        
    return df_onset






        
       
        
        
