import pandas as pd
import numpy as np

from .constants import glucose_levels, mmoll_mgdl

def minimum_eventtime(X:pd.Series,  mintime:str):
    """
    Set event to false if minimum event time is violated
    Note: a pandas rolling function does not work here, as it would not take into account the entire event

    X       - pd.Series with dtype int indicating event (1 or 0)
    mintime - minimum even time
    """
    for start, end in zip(X[X.diff() == 1].index, X[X.diff().shift(-1) == -1].index):        
        if end - start < pd.to_timedelta(mintime):
            X.loc[(X.index >= start) & (X.index <= end)] = 0
    return X

def hypo(X:pd.Series, mintime:str='14min45S', unit:str='mgdl'):
    """
    Calculate hypo according to definition of https://doi.org/10.2337/dc17-1600
    Note: make sure your data has timestamp indices
    """
    if unit == 'mmoll':
        res = (X * mmoll_mgdl < glucose_levels['target'][0]).astype(int)
    elif unit == 'mgdl':
        res = (X < glucose_levels['target'][0]).astype(int)
    res = minimum_eventtime(res, mintime)
    return res

def hyper(X:pd.Series, mintime:str='14min45S', unit:str='mgdl'):
    """
    Calculate hyper according to definition of https://doi.org/10.2337/dc17-1600
    Note: make sure your data has timestamp indices
    """
    if unit == 'mmoll':
        res = (X * mmoll_mgdl > glucose_levels['target'][1]).astype(int)
    elif unit == 'mgdl':
        res = (X > glucose_levels['target'][1]).astype(int)
    res = minimum_eventtime(res, mintime)
    return res

def symmetric_scale(X:pd.Series, unit:str='mgdl'):
    # symmetric scaling for blood glucose
    if unit == 'mgdl':
        return 1.509*(np.log(X)**1.084 - 5.381)
    elif unit == 'mmoll':
        return 1.794*(np.log(X)**1.026 - 1.861)

def LBGI(X:pd.Series):
    # https://doi.org/10.2337/diacare.21.11.1870
    return symmetric_scale(X).apply(lambda x: 10*x**2 if x < 0 else 0).mean()

def HBGI(X:pd.Series):
    # https://doi.org/10.2337/dc06-1085
    return symmetric_scale(X).apply(lambda x: 10*x**2 if x >= 0 else 0).mean()

def time_in_level(X:pd.Series, l:str, levels:dict=glucose_levels):
    """
    Return number of measurements within glucose range l
    """
    return ((X >= levels[l][0]) & (X <= levels[l][1])).sum()

def perc_in_level(X:pd.Series, l:str, levels:dict=glucose_levels):
    """
    Return percentage of measurements within glucose range l
    """
    return ((X >= levels[l][0]) & (X <= levels[l][1])).sum() / X.count() * 100

def stats_cgm(X:pd.DataFrame, col:str='Glucose Value (mg/dL)'):
    return {'time_in_hypo'     : time_in_level(X[col], 'hypo'),
            'time_in_hypoL2'   : time_in_level(X[col], 'hypo L2'),
            'time_in_hypoL1'   : time_in_level(X[col], 'hypo L1'),
            'time_in_target'   : time_in_level(X[col], 'target'),
            'time_in_hyper'    : time_in_level(X[col], 'hyper'),
            'time_in_hyperL1'  : time_in_level(X[col], 'hyper L1'),
            'time_in_hyperL2'  : time_in_level(X[col], 'hyper L2'),
            'perc_in_hypo'     : perc_in_level(X[col], 'hypo'),
            'perc_in_hypoL2'   : perc_in_level(X[col], 'hypo L2'),
            'perc_in_hypoL1'   : perc_in_level(X[col], 'hypo L1'),
            'perc_in_target'   : perc_in_level(X[col], 'target'),
            'perc_in_hyper'    : perc_in_level(X[col], 'hyper'),
            'perc_in_hyperL1'  : perc_in_level(X[col], 'hyper L1'),
            'perc_in_hyperL2'  : perc_in_level(X[col], 'hyper L2'),
            'glucose_mean'     : X[col].mean(),
            'glucose_std'      : X[col].std(),
            'glucose_cv'       : X[col].std() / X[col].mean() * 100,
            'glucose_rate'     : X['glucose_rate'].mean(),
            'completeness'     : X[col].count() / X['timestamp'].count(),
            'count'            : X[col].count(),
            'LBGI'             : LBGI(X[col]),
            'HBGI'             : HBGI(X[col]),
            'AUC'              : np.trapz(y=X[col], x=X['timestamp']) / np.timedelta64(5, 'm'),
            'hypo'             : X['hypo'].any(),
            'hyper'            : X['hyper'].any()}