"""
This module for implementing a Maxwell-Boltzmann distribution in the Galactic frame
that accounts for the time-dependent motion of the Earth relativeto the Sun.
"""

from axionpy import units as u 
from axionpy.velocity import _vobs

import numpy as np
import astropy.coordinates as coord
from astropy.time import Time

# default distribution parameters
_sigma = u.toNaturalUnits(220.0*u.km/u.s).to_value(u.dimensionless_unscaled)

def maxwell_boltzmann_distribution(w):
    """
    Maxwell-Boltzmann Galactic-frame velocity distribution function.
    This is the default velocity distribution function.

    Parameters
    ------------
    w : (3,...) array-like
        Galactic-frame velocity coordinates in (u,v,w) basis
        in dimensionless units.

    Returns
    ------------
    f : array-like
        Galactic-frame elocity distribution function.
        Shape must satisfy (3,) + np.shape(f) == np.shape(w)
    """
    w2 = w[0]**2 + w[1]**2 + w[2]**2
    return np.exp(-0.5*w2/_sigma**2)/(2.*np.pi*_sigma**2)**(3./2.)

def correlator(mode, m, t1, t2, **kwargs):
    """
    Computes the specified two-point correlation function. With rhodm=1.
    In these units, the output is dimensionless.

    Parameters
    --------------------
    mode : str 
           Must be of the form 'XiYj'
           where X = 'A' or 'B'
           and i = 'x', 'y', or 'z'

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

    f : (optional) callable
        Galactic-frame velocity distribution function that takes the
        velocity in dimenionless units as argument.
        Must have call signature f(w) where w is a (3,...) arbitrary ndarray.
        Must return f such that (3,) + np.shape(f) == np.shape(w)
        Defaults to a Maxwell-Boltzmann distribution with dispersion of 220 km/s

    vobs_1 : (optional) astropy.Quantity array-like
             The velocity of the observer at times t1 in (u,v,w) coordinates.
             Satisfyies np.shape(vobs_1) == (3,)+np.shape(t1)

    vobs_2 : (optional) astropy.Quantity array-like
             The velocity of the observer at times t2 in (u,v,w) coordinates.
             Satisfyies np.shape(vobs_2) == (3,)+np.shape(t2)

    epoch : (optional) str or astropy.Time
            The epoch at which to evaluate the time series.
            Must be specified if vobs_1 or vobs_2 are not specified.

    loc : (optional) astropy.EarthLocation
          The location of the lab relative to Earth.
          The lab location must be specified if vobs_1 or vobs_2 are not specified.
          The lab location can be specified either with loc or with lat, lon
    
    lat, lon : (optional) float or astropy.Quantity (angle)
               The location of the lab (latitude and longitude) relative to Earth.
               If specified as a float, it is interpreted as an angle in degrees.
               The lab location must be specified if vobs_1 or vobs_2 are not specified.
               The lab location can be specified either with loc or with lat, lon
    """

    if 'f' in kwargs:
        f = kwargs['f']
    else:
        f = maxwell_boltzmann_distribution
        
    if 'vobs_1' in kwargs:
        vobs_1 = kwargs['vobs_1']
    else:
        vobs_1 = None

    if 'vobs_2' in kwargs:
        vobs_2 = kwargs['vobs_2']
    else:
        vobs_2 = None
        
    if vobs_1 is None or vobs_2 is None:
        if epoch in kwargs:
            epoch  = kwargs['epoch']
            if not isinstance(epoch, Time):
                epoch = Time(epoch)
        else:
            raise Exception("ERROR: must specify epoch if vobs_1 or vobs_2 not specified")

        if 'loc' in kwargs:
            loc = kwargs['epoch']
        elif 'lat' in kwargs and 'lon' in kwargs:
            loc = coord.EarthLocation(lat=kwargs['lat'], lon=kwargs['lon'])
        else:
            raise Exception("ERROR: must specify lab location if vobs_1 or vobs_2 not specified.")

        if vobs_1 is None:
            vobs_1 = _vobs(loc, epoch+t1)
        if vobs_2 is None:
            vobs_2 = _vobs(loc, epoch+t2)
        
    vmax = 1000.*u.km/u.s
    
    # convert velocities to dimensionless units
    vmax = u.convert(vmax, u.dimenionless_unscaled, value=True)
    vobs_1 = u.convert(vobs_1, u.dimensionless_unscaled, value=True)
    vobs_2 = u.convert(vobs_2, u.dimensionless_unscaled, value=True)

    
