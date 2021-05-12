"""
This module for the "1-dimensional" stochastic time domain analysis.

It works the same as the "full" time domain analysis except the model assumes
that the axion gradient is *always* in the direction of vsun
and the axion "speed" is *always* vo.
"""
import numpy as np 
import astropy.units as u
import natural_units as nat
import scipy.optimize as opt
from scipy import stats
import ..maxwell as maxwell
import ..axis as axs
from ..constants import _rhodm
import ..matrix as matrix

def cov(m, t, **kwargs):
    """
    m : astropy.Quantity
    t : (N,) astropy.Quantity
    
    components : (optional) (3,N) array of axis components
    axis : (optional)
    lat, lon, theta, phi : (optional)

    kwargs : passed to Axis.components if components is not specified
    
    Return CovMatrix object
    """
    if 'components' in kwargs:
        components = kwargs['components']
    elif 'axis' in kwargs:
        axis = kwargs['axis']
        components = axis.components(t=t, basis='xyz', **kwargs)
    elif 'lat' in kwargs and 'lon' in kwargs and 'theta' in kwargs and 'phi' in kwargs:
        axis = axs.Axis(kwargs['lat'], kwargs['lon'], kwargs['theta'], kwargs['phi'])
        components = axis.components(t=t, basis='xyz', **kwargs)
    else:
        raise Exception("ERROR: must specify components, axis or (lat, lon, theta, phi)")
    
    t1, t2 = np.meshgrid(t, t, indexing='ij')
    mz_1, mz_2 = np.meshgrid(components[2], components[2], indexing='ij')

    AzAz = maxwell.correlator('AzAz', m, t1, t2, **kwargs)
    AzBz = maxwell.correlator('AzBz', m, t1, t2, **kwargs)

    AA = mz_1*mz_2*AzAz
    AB = mz_1*mz_2*AzBz

    C = np.block([[AA, AB], [-AB, AA]])

    return matrix.CovMatrix(C)

# use identical functions from full analysis
from time_domain import compute_coefficients, loglikelihood, maximize_likelihood, profile_likelihood, frequentist_upper_limit
        
