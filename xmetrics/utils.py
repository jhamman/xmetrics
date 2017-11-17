import numpy as np
import scipy
import numba


def _scipy_fit_and_ppf(vals, dist=scipy.stats.gamma, pone=None, fit_args=None):
    if fit_args is None:
        fit_args = {}
    shape1, loc1, scale1 = dist.fit(vals, **fit_args)
    return dist.ppf(q=pone, a=shape1, loc=loc1, scale=scale1)


@numba.jit(nopython=True, cache=True)
def spatial_autocorrelation(data, minlag=1, maxlag=50, timelags=4):
    '''Ethan's spatial autocorrelation function, just reformatted a bit'''
    shape = data.shape
    out_shape = (shape[0], shape[1], maxlag - minlag + 1 + timelags + 1)
    rs = np.full(out_shape, np.nan)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not np.isnan(data[i, j, 0]):
                for lag in range(minlag, min(shape[0] - i - 1,
                                             shape[1] - j - 1,
                                             maxlag + 1)):
                    r = 0.0
                    n = 0.0
                    # check for non-fill locations
                    if not np.isnan(data[i + lag, j, 0]):
                        r2 = corr(data[i, j], data[i + lag, j])
                        r += r2
                        n += 1
                    if not np.isnan(data[0, i, j + lag]):
                        r4 = corr(data[i, j], data[i, j + lag])
                        r += r4
                        n += 1

                    if n > 0:
                        rs[i, j, lag - 1] = r / n

                for t in range(1, timelags):
                    r = corr(data[i, j, t:], data[i, j, :-t])
                    rs[i, j, maxlag + t] = r
    return rs


@numba.jit(nopython=True, cache=True)
def corr(data1, data2):
    '''https://stackoverflow.com/a/29194624/1757464'''
    n = data1.size

    sum1 = 0.
    sum2 = 0.
    for i in range(n):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / n
    mean2 = sum2 / n

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in range(n):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / n) ** .5
    std2 = (var_sum2 / n) ** .5
    cross_mean = cross_sum / n

    return (cross_mean - mean1 * mean2) / (std1 * std2)
