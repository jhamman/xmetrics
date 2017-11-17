from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import xarray as xr

from xarray.testing import assert_equal

import xmetrics

from . import raises_regex


np.random.seed(3)


@pytest.fixture(scope="module")
def da_lat_lon():
    return xr.DataArray(np.random.random((10, 20)), dims=('lat', 'lon'),
                        coords={'lat': np.linspace(-90, 90, 10, endpoint=True),
                                'lon': np.linspace(0, 360, 20)})


@pytest.fixture(scope="module")
def da_time_lat_lon():
    ntimes = 365
    return xr.DataArray(np.random.random((ntimes, 10, 20)),
                        dims=('time', 'lat', 'lon'),
                        coords={'time': np.arange(ntimes),
                                'lat': np.linspace(-90, 90, 10, endpoint=True),
                                'lon': np.linspace(0, 360, 20)})


def test_wetdry_metric(da_time_lat_lon):
    ds = da_time_lat_lon.xmetrics.wetdry(threshold=0.1)
    assert isinstance(ds, xr.Dataset)
    assert 'wet_fraction' in ds
    assert 'wetspell_length' in ds
    assert 'dryspell_length' in ds


def test_fit_dist_metric(da_time_lat_lon):
    actual1 = da_time_lat_lon.xmetrics.fit_dist('gamma', [0.01, 0.5, 0.999],
                                                dim='time')
    actual2 = da_time_lat_lon.xmetrics.fit_dist('gamma', 0.5, dim='time')

    assert_equal(actual1.sel(pone=0.5), actual2.sel(pone=0.5))

    with raises_regex(ValueError, 'dim is a required argument'):
        da_time_lat_lon.xmetrics.fit_dist('gamma', 0.5)


def test_histogram_metric(da_time_lat_lon):
    actual = da_time_lat_lon.xmetrics.histogram()
    assert isinstance(actual, xr.DataArray)


def test_spatial_autocorrelations(da_time_lat_lon):
    actual = da_time_lat_lon.xmetrics.spatial_autocorrelations()
    assert isinstance(actual, xr.DataArray)
    assert 'lag' in actual.dims
