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
            precip threshold to use as definition of a wet day

        Returns
        -------
        wet_fraction : DataArray
            Wet day fraction
        wetspell_length : DataArray
            Mean wetspell length
        dryspell_length: DataArray
            Mean dryspell length
        '''

        ntimes = len(self._obj['time'])

        wetdays = xr.zeros_like(self._obj)
        wetdays = (self._obj > threshold)
        total_wetdays = wetdays.sum(dim='time')

        wet_fraction = total_wetdays / ntimes

        diff_wetdays = wetdays.diff('time', n=1)
        wetspells = (diff_wetdays == 1).sum(dim='time')
        dryspells = (diff_wetdays == -1).sum(dim='time')

        # wetdays should be 0 so length will be 0/1
        wetspells = wetspells.where(wetspells == 0, 1)
        wetspell_length = total_wetdays / wetspells

        # drydays should be 0 so length will be 0/1
        dryspells = dryspells.where(dryspells == 0, 1)
        dryspell_length = (ntimes - total_wetdays) / dryspells

        return wet_fraction, wetspell_length, dryspell_length

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
                 dim=None, axis=None, fit_args=None):
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
        axis : int
            axis for which to fit the distribution over
        fit_args : dict
            Dictionary of arguments to pass to `distribution.fit`

        Returns
        -------
        fitted : DataArray
            Fitted values at the points defined by the non-exceedance
            probabilities
       '''

        from .utils import _scipy_fit_and_ppf

        nonexceedance_probs = np.atleast_1d(nonexceedance_probs)

        if not isinstance(distribution, scipy.stats.rv_continuous):
            distribution = getattr(scipy.stats, distribution)

        if axis is None:
            axis = self._obj.get_axis_num(dim)
        elif dim is None:
            dim = self._obj.dims[axis]
        else:
            return ValueError('must provide either dim or axis')

        # fit distribution and extract values at points defined by
        # `nonexceedance_probs`
        out = np.apply_along_axis(_scipy_fit_and_ppf, axis, self._obj,
                                  pone=nonexceedance_probs, fit_args=fit_args)

        # dims
        new_dims = list(self._obj.dims)
        new_dims[axis] = 'prob_non_exceedance'

        # coords
        new_coords = dict(self._obj.coords)
        new_coords.pop(dim)
        new_coords['prob_non_exceedance'] = nonexceedance_probs

        return xr.DataArray(out, dims=new_dims, coords=new_coords)

    def spatial_autocorrelations(self, **kwargs):
        '''Calculate the spatial autocorrelation

        Parameters
        ----------
        kwargs : optional
            Additional arguments to utils.spatial_autocorrelation function
        '''
        from .utils import spatial_autocorrelation

        assert self._obj.get_axis_num('time') == 0

        data = self._obj.data

        r = spatial_autocorrelation(data, **kwargs)

        r = xr.DatArray(r, dims=('lag', ))

        return r
