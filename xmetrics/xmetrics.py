from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import OrderedDict

import dask.array
import xarray as xr
import numpy as np
import scipy.stats

from xarray.core.pycompat import dask_array_type


@xr.register_dataarray_accessor('xmetrics')
class XMetrics(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def wetdry(self, threshold=0):
        '''Calculate wet/dry statistics for data given a precip threshold

        Parameters
        ----------
        threshold : float
            Precipitation threshold to use as definition of a wet day

        Returns
        -------
        metrics : Dataset
            Dataset with three metric variables: `wet_fraction`,
            `wetspell_length`, `dryspell_length`
        '''

        from .metrics import wetdry

        return wetdry(self._obj, threshold=threshold)

    def histogram(self, precip=True, bins=50, hist_range=None, **kwargs):
        '''Calculate a histogram for data

        Parameters
        ----------
        precip : bool
            For precip calculate on a log scale.
        bins : int or sequence of scalars or str, optional
            See numpy.histogram for description and options.
        hist_range : (float, float), optional
            Equivalent to numpy.histogram range keyword arguemnt.
        kwargs : optional
            Additional arguments to wrapped numpy.histogram function

        Returns
        -------
        hist : DataArray
            The values of the histogram. The bin-centers are included as a
            coordinate variable

        See Also
        --------
        numpy.histogram
        dask.array.histogram
        '''

        flat_data = self._obj.data.flat

        if isinstance(self._obj.data, dask_array_type):
            array_mod = dask.array
        else:
            array_mod = np

        if precip:
            flat_data = flat_data[flat_data > 0]
            usedata = array_mod.log10(flat_data)
            if hist_range is None:
                hist_range = (-2, 3)
        else:
            usedata = flat_data
            if hist_range is None:
                hist_range = (-30, 60)

        hist, bin_edges = array_mod.histogram(usedata, bins=bins,
                                              range=hist_range, **kwargs)

        if precip:
            bin_edges = 10.0**bin_edges

        # return bin centers instead of edges
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return xr.DataArray(hist, dims='bins', coords={'bins': bin_centers})

    def fit_dist(self, distribution, nonexceedance_probs,
                 dim=None, fit_args=None):
        '''Fit scipy.stats distribution and return values at specified
           non-exceedance probabilities

        Parameters
        ----------
        distribution : str or scipy.stats.rv_continuous
            A continuous random variable distribution to fit to this object's
            data
        nonexceedance_probs : array_like
            probability of non-exceedance
        dim : str
            dimension name for which to fit the distribution over
        fit_args : dict
            Dictionary of arguments to pass to `distribution.fit`

        Returns
        -------
        fitted : DataArray
            Fitted values at the points defined by the non-exceedance
            probabilities
       '''

        from .utils import _scipy_fit_and_ppf
        if dim is None:
            raise ValueError('dim is a required argument')

        nonexceedance_probs = np.atleast_1d(nonexceedance_probs)
        if nonexceedance_probs.ndim > 1:
            raise ValueError('Rank of nonexceedance_probs (%s) is too '
                             'large' % nonexceedance_probs.ndim)

        if not isinstance(distribution, scipy.stats.rv_continuous):
            distribution = getattr(scipy.stats, distribution)

        dims = OrderedDict(zip(self._obj.dims, self._obj.shape))
        del dims[dim]
        dims['pone'] = len(nonexceedance_probs)

        # fit distribution and extract values at points defined by
        # `nonexceedance_probs`
        func = functools.partial(_scipy_fit_and_ppf, dist=distribution,
                                 pone=nonexceedance_probs, fit_args=fit_args)
        out = xr.apply_ufunc(func, self._obj,
                             vectorize=True,
                             input_core_dims=[[dim]],
                             output_core_dims=[['pone']],
                             output_dtypes=[np.float],
                             output_sizes=dims,
                             dask='parallelized')

        out.coords['pone'] = xr.Variable('pone', nonexceedance_probs)

        return out

    def spatial_autocorrelations(self, **kwargs):
        '''Calculate the spatial autocorrelation

        Parameters
        ----------
        kwargs : optional
            Additional arguments to utils.spatial_autocorrelation function
        '''
        from .utils import spatial_autocorrelation

        xy_dims = list(self._obj.dims)
        xy_dims.remove('time')
        trans_dims = xy_dims + ['time']
        new_dims = ['lag'] + xy_dims

        da = self._obj.transpose(*trans_dims)

        r = spatial_autocorrelation(da.data, **kwargs)

        r = xr.DataArray(r, dims=new_dims, name='spatial_autocorrelation')

        return r
