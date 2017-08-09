import pytest
import numpy as np
import xarray as xr

import xmetrics

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
    da_time_lat_lon.xmetrics.wetdry(threshold=0.1)


def test_fit_dist_metric(da_time_lat_lon):
    da_time_lat_lon.xmetrics.fit_dist('gamma', [0.01, 0.999], dim='time')
    da_time_lat_lon.xmetrics.fit_dist('gamma', 0.5, dim='time')


def test_histogram_metric(da_time_lat_lon):
    da_time_lat_lon.xmetrics.histogram()


def spatial_autocorrelations(da_time_lat_lon):
    da_time_lat_lon.xmetrics.spatial_autocorrelations()
