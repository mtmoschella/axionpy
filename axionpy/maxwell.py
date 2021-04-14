"""
This module for implementing a Maxwell-Boltzmann distribution
"""

import natural_units as nat
import astropy.units as u
import astropy.constants as const

# default distribution parameters
_sigma = nat.toNaturalUnits(220.0*u.km/u.s, value=True)
_vo = nat.toNaturalUnits(234.0*u.km/u.s, value=True)

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
    Computes the specified two-point correlation function.

    mode : str in {'AxAx', 'AxBx',
                   'AyAy', 'AyBy',
                   'AzAz', 'AzBz',
                   'AA', 'AB'}
           Specifies which correlators to compute.
           
    m : astropy.Quantity
        The axion mass, must have units equivalent to frequency

    t1 : array-like
         The times at which to evaluate the first operator.
         Must be same shape as t2.

    t2 : array-like
         The times at which to evaluate the second operator.
         Must be same shape as t1.
    """

    if mode=='AxAx':
        pass
    elif mode=='AxBx':
        pass
    elif mode=='AyAy':
        pass
    elif mode=='AyBy':
        pass
    elif mode=='AzAz':
        pass
    elif mode=='AzBz':
        pass
    elif mode=='AA':
        pass
    elif mode=='BB':
        pass
    else:
        raise Exception("ERROR: unrecognized mode "+str(mode)+" for correlator")
    
