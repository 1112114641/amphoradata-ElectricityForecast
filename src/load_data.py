"""
Load all necessary data for modelling.
"""
import pandas as pd
from math import floor
import numpy as np
from datetime import datetime, timezone, timedelta
import math
import time
from src import APIfetch
import src.tool as tool



def load_data():
    """
    loads necessary data from amphoradata.com
    """
    params = {'id': '',
        'start_time': datetime(2019,11,9,11),#YYYY,MM,DD,hh,mm
        'end_time': datetime.now(),
        #'filter': ''
        }

    e_ids = {'electricity_NSW': 'ecc5263e-83b6-42d6-8852-64beffdf204e',
            'electricity_QLD': 'ef22fa0f-010c-4ab1-8a28-a8963f838ce9',
            'electricity_VIC': '3b66da5a-0723-4778-98fc-02d619c70664',
            'electricity_SA': '89c2e30d-78c8-46ef-b591-140edd84ddb6'}

    electricity_dict = {}
    for _ in e_ids:
        params['id'] = e_ids[_]
        data_tempe = APIfetch.fetch_data(params)
        # to prevent the API from returning only a subset of the desired
        # data, this check ensures that the returned data is <max
        assert data_tempe.shape[0]<=10000, 'API maximum row number surpassed'
        if 'filter' in params:
            del data_tempe['periodType.{}'.format(params['filter'])]
        else:
            del data_tempe['periodType']

        # add the _state suffix to column names:
        data_tempe.columns = [col +'_'+_[_.index('_')+1:] for col in data_tempe.columns]
        data_tempe = data_tempe.sort_index()

        # prevent some weird pandas 'feature' from resampling the df:
        data_tempe = data_tempe.apply(pd.to_numeric, errors='coerce')
        electricity_dict[_] = data_tempe.resample('60T').mean()

    forecast_dict = {}
    for _ in e_ids:
        params['filter'] = 'Forecast'
        params['id'] = e_ids[_]
        data_tempf = APIfetch.fetch_data(params)
        del data_tempf['periodType.Forecast']

        # add the _state suffix to column names:
        data_tempf.columns = [col +'_'+_[_.index('_')+1:] for col in data_tempf.columns]
        data_tempf = data_tempf.sort_index()

        # prevent some weird pandas 'feature' from resampling the df:
        data_tempf = data_tempf.apply(pd.to_numeric,                       errors='coerce')
        forecast_dict[_] = data_tempf.resample('60T').mean()
        del params['filter']

    w_ids = {'weather_NSW': '11fd3d6a-12e4-4767-9d52-03271b543c66',
            'weather_QLD': 'a46f461f-f7ee-4cc5-a1e4-569960ea5ed8',
            'weather_VIC': 'd48ac35f-c658-41c1-909a-f662d6f3a972',
            'weather_SA': 'f860ba45-9dda-41e0-91aa-73901a323318'}

    weather_dict = {}
    for _ in w_ids:
        params['id'] = w_ids[_]
        data_tempw = APIfetch.fetch_data(params)

        if 'filter' in params:
            del data_tempw['description.{}'.format(params['filter'])]
        else:
            del data_tempw['description']

        # add the _state suffix to column names:
        data_tempw.columns = [col +'_'+_[_.index('_')+1:] for col in data_tempw.columns]
        data_tempw = data_tempw.sort_index()

        # prevent some weird pandas 'feature' from resampling the df:
        data_tempw = data_tempw.apply(pd.to_numeric,                       errors='coerce')
        weather_dict[_] = data_tempw.resample('60T').mean()

    # join all electricity data
    df_elec = electricity_dict['electricity_NSW'].join(electricity_dict['electricity_QLD'])
    df_elec = df_elec.join(electricity_dict['electricity_VIC'])
    df_elec = df_elec.join(electricity_dict['electricity_SA'])
    df_elec = tool.create_diffs(df_elec, list(range(len(df_elec.columns))))

    # join all fore data
    df_fore = forecast_dict['electricity_NSW'].join(forecast_dict['electricity_QLD'])
    df_fore = df_fore.join(forecast_dict['electricity_VIC'])
    df_fore = df_fore.join(forecast_dict['electricity_SA'])
    df_fore = tool.create_diffs(df_fore, list(range(len(df_fore.columns))))

    # outer join for all weather data
    df_weather = weather_dict['weather_NSW'].join(weather_dict['weather_QLD'])
    df_weather = df_weather.join(weather_dict['weather_VIC'])
    df_weather = df_weather.join(weather_dict['weather_SA'])
    df_weather = tool.create_diffs(df_weather, list(range(len(df_weather.columns))))


    # re-sort dataset columns and extract test data y
    df_elec = df_elec.reindex(sorted(df_elec.columns), axis=1)
    y = df_elec.iloc[:,8:12]
    for _ in y.columns:
        del df_elec[_]

    print('elec shape: ',df_elec.shape)
    print('y shape: ',y.shape)

    # Create complete dataset of all train variables
    df_all = df_elec.join(df_weather)
    df_all = df_all.join(df_fore)

    print('df_all shape: ',df_all.shape)

    #also atomise the datetime index, will be v useful for large-ish datasets, as spikes on e.g. weekends, become identifiable for the model:
    # split dates into year month day day of week hour, etc for additional features
    df_all = tool.split_dates_df(df_all)

    # to save and/or stitch previously downloaded data csv's:
    df_all.to_csv('./2_raw_data/df_all-to-7thJan1.csv',
                header=True,
                index=True)
    y.to_csv('./2_raw_data/y-to-7thJan1.csv',
            header=True,
            index=True)
    # ensure equal shape on both X & Y, so both train and test data share the same timeframe
    df_all = df_all.interpolate(method='spline',
                                order=3,
                                limit_direction='forward',
                                axis=0)
    df_all = df_all.dropna()
    y = y.dropna()

    # ensuring both are consistent:
    y = y[y.index.isin(df_all.index)]
    df_all = df_all[df_all.index.isin(y.index)]
    y = y[y.index.isin(df_all.index)]
    #y = y[(y.index>=df_all.index[0]) & (y.index<=df_all.index[-1])]
    assert all(dat in df_all.index for dat in y.index), 'Index differs for df_all and y!'

    print('df_all.shape: ',df_all.shape)
    print('y.shape: ',y.shape)

    return df_all, y