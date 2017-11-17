from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import xarray as xr


def metric_time_mean(da):
    return da.mean(dim='time')


def metric_time_max(da):
    return da.max(dim='time')


def metric_time_min(da):
    return da.min(dim='time')


def metric_time_wdf(da, threshold=0.):
    return (da > threshold).mean(dim='time')


def metric_time_std(da):
    return da.resample(dim='time', how='mean').std(dim='time')


def metric_spatial_mean(da, dims=('lon', 'lat')):
    return da.mean(dim=dims)


def metric_spatial_max(da, dims=('lon', 'lat')):
    return da.max(dim=dims)


def wetdry(da, threshold=0):
    '''Calculate wet/dry statistics for data given a precip threshold

    Parameters
    ----------
    da : DataArray
        Precipitation array with time dimension
    threshold : float
        Precipitation threshold to use as definition of a wet day

    Returns
    -------
    metrics : Dataset
        Dataset with three metric variables: `wet_fraction`, `wetspell_length`,
        `dryspell_length`
    '''

    ntimes = len(da['time'])

    wetdays = (da > threshold)
    total_wetdays = wetdays.sum(dim='time')

    # start of wetspells == 1, dryspells == -1, no change == 0
    diff_wetdays = wetdays.astype(np.int).diff('time', n=1)

    # pick out the begining of wet/dry spells
    wetspells = (diff_wetdays == 1).sum(dim='time')
    wetspells += wetdays.isel(time=0)
    dryspells = (diff_wetdays == -1).sum(dim='time')
    dryspells += (wetdays.isel(time=0) == 0)

    ds = xr.Dataset({'wet_fraction': total_wetdays / ntimes,
                     'wetspell_length': total_wetdays / wetspells,
                     'dryspell_length': (ntimes - total_wetdays) / dryspells})

    return ds
