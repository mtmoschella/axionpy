"""
This module for implementing a Maxwell-Boltzmann distribution
"""

from axionpy import units as u 
from axionpy import axis as axs
import numpy as np

# default distribution parameters
_sigma = u.toNaturalUnits(220.0*u.km/u.s).to_value(u.dimensionless_unscaled)
_vo = u.toNaturalUnits(234.0*u.km/u.s).to_value(u.dimensionless_unscaled)

def _A_par(x, **kwargs):
    """
    Helper function.
    """
    
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = _sigma

    if 'vo' in kwargs:
        vo = kwargs['vo']
    else:
        vo = _vo
        
    prefactor = np.sqrt((sigma**2 + vo**2)**2 + sigma**4*x**2)
    exponent = np.exp(-0.5*(vo**2/sigma**2)*x**2/(1+x**2))
    poly = 1./(1+x**2)**(7./4.)
    return prefactor*exponent*poly
    

def _A_perp(x, **kwargs):
    """
    Helper function.
    """
    
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = _sigma

    if 'vo' in kwargs:
        vo = kwargs['vo']
    else:
        vo = _vo
    
    prefactor = sigma**2
    exponent = np.exp(-0.5*(vo**2/sigma**2)*x**2/(1+x**2))
    poly = 1./(1+x**2)**(5./4.)
    return prefactor*exponent*poly

def _Psi_par(x, **kwargs):
    """
    Helper function.
    """
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = _sigma

    if 'vo' in kwargs:
        vo = kwargs['vo']
    else:
        vo = _vo
    
    return (0.5*vo**2/sigma**2)*x/(1+x**2) + 0.5*7.*np.arctan(x) - np.arctan(sigma**2*x/(sigma**2 + vo**2))

def _Psi_perp(x, **kwargs):
    """
    Helper function.
    """
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = _sigma

    if 'vo' in kwargs:
        vo = kwargs['vo']
    else:
        vo = _vo
        
    return (0.5*vo**2/sigma**2)*x/(1+x**2) + 0.5*5.*np.arctan(x)

def correlator(mode, m, t1, t2, **kwargs):
    """
    Computes the specified two-point correlation function. With rhodm=1.
    In these units, the output is dimensionless.

    Parameters
    --------------------
    mode : str in {'AxAx', 'AxBx',
                   'AyAy', 'AyBy',
                   'AzAz', 'AzBz'}
           Specifies which correlators to compute.
           The convention for the off diagonal elements is that
           'AxBx' specifies the correator <Ax(t1)*Bx(t2)>
           where S(t) = (A*cos(m*t) - B*sin(m*t))*mhat_dot_xhat + ...
           (Note the sign convention of the B coefficient).
           
    m : astropy.Quantity
        The axion mass, must have units equivalent to frequency

    t1 : astropy.Quantity array-like
         The times at which to evaluate the first operator.
         Must be same shape as t2.

    t2 : astropy.Quantity array-like
         The times at which to evaluate the second operator.
         Must be same shape as t1.
    """

    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = _sigma
        
    x = (m*(t2-t1)*sigma**2).to_value(u.dimensionless_unscaled)
    
    if mode=='AxAx':
        return _A_perp(x, **kwargs)*np.cos(_Psi_perp(x,**kwargs))
    elif mode=='AxBx':
        return _A_perp(x, **kwargs)*np.sin(_Psi_perp(x,**kwargs))
    elif mode=='AyAy':
        return _A_perp(x, **kwargs)*np.cos(_Psi_perp(x,**kwargs))
    elif mode=='AyBy':
        return _A_perp(x, **kwargs)*np.sin(_Psi_perp(x,**kwargs))
    elif mode=='AzAz':
        return _A_par(x, **kwargs)*np.cos(_Psi_par(x,**kwargs))
    elif mode=='AzBz':
        return _A_par(x, **kwargs)*np.sin(_Psi_par(x,**kwargs))
    else:
        raise Exception("ERROR: unrecognized mode "+str(mode)+" for correlator")
    
