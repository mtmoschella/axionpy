"""
This module for implementing an arbitrary velocity distribution in the Galactic frame
that accounts for the time-dependent motion of the Earth relativeto the Sun.
"""

from axionpy import units as u 
from axionpy.velocity import vcirc, _vobs, _x, _y, _z
import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
from tqdm import tqdm

_basis_vectors = {'x' : _x, 'y' : _y, 'z' :_z}

### default distribution parameters
# Maxwell-Boltzmann velocity dispersion
_sigma = u.toNaturalUnits((vcirc/np.sqrt(2.))*u.km/u.s).to_value(u.dimensionless_unscaled)

def maxwell_boltzmann_distribution(w):
    """
    Maxwell-Boltzmann Galactic-frame velocity distribution function.
    This is the default velocity distribution function.

    Parameters
    ------------
    w : (...,3) array-like
        Galactic-frame velocity coordinates in (u,v,w) basis
        in dimensionless units.

    Returns
    ------------
    f : array-like
        Galactic-frame elocity distribution function.
        Shape must satisfy np.shape(f) + (3,) == np.shape(w)
    """
    assert np.shape(w)[-1]==3, "ERROR: w must have shape (...,3)"
    w2 = w[...,0]**2 + w[...,1]**2 + w[...,2]**2
    return np.exp(-0.5*w2/_sigma**2)/(2.*np.pi*_sigma**2)**(3./2.)

def correlator(mode, m, t1, t2, **kwargs):
    """
    Computes the specified two-point correlation function. With g=rhodm=1.
    In these units, the output is dimensionless.

    Parameters
    --------------------
    mode : str 
           Must be of the form 'XiYj'
           where X = 'A' or 'B'
           and i = 'x', 'y', or 'z'

           Specifies which correlators to compute.
           The convention for the off diagonal elements is that
           'AxBx' specifies the correlator <Ax(t1)*Bx(t2)>
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
             Satisfyies np.shape(vobs_1) == np.shape(t1)+(3,)

    vobs_2 : (optional) astropy.Quantity array-like
             The velocity of the observer at times t2 in (u,v,w) coordinates.
             Satisfyies np.shape(vobs_2) == np.shape(t2)+(3,)

    epoch : (optional) str or astropy.Time
            The epoch at which to evaluate the time series.
            Only used if vobs_1 or vobs_2 are not specified.
            Defaults to J2000.

    loc : (optional) astropy.EarthLocation
          The location of the lab relative to Earth.
          The lab location must be specified if vobs_1 or vobs_2 are not specified.
          The lab location can be specified either with loc or with lat, lon
    
    lat, lon : (optional) float or astropy.Quantity (angle)
               The location of the lab (latitude and longitude) relative to Earth.
               If specified as a float, it is interpreted as an angle in degrees.
               The lab location must be specified if vobs_1 or vobs_2 are not specified.
               The lab location can be specified either with loc or with lat, lon

    verbose : (optional) bool
              Run in verbose mode.
              Defaults to False.

    Returns
    --------------
    corr : numpy.ndarray
           The specified two-point correlator of shape np.shape(t1)
    """

    t_shape = np.shape(t1)
    if np.shape(t2)!=t_shape:
        raise Exception("ERROR: t1 and t2 must have the same shape")
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    nt = len(t1_flat)

    # unpack mode
    if len(mode)!=4:
        raise Exception("ERROR: invalid mode "+str(mode))
    X1, i, X2, j = mode
    ihat = _basis_vectors[i]
    jhat = _basis_vectors[j]
    if X1==X2:
        trig = np.cos
    elif X1=='A' and X2=='B':
        trig = np.sin
    else:
        trig = lambda x : -1.*np.sin(x)
    
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
        if 'epoch' in kwargs:
            epoch  = kwargs['epoch']
            if not isinstance(epoch, Time):
                epoch = Time(epoch)
        else:
            epoch = Time('J2000.0')

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

    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
        
    # convert velocities to dimensionless units
    vobs_1 = u.convert(vobs_1, u.dimensionless_unscaled, value=True).reshape((nt,3)) # (nt,3)
    vobs_2 = u.convert(vobs_2, u.dimensionless_unscaled, value=True).reshape((nt,3)) # (nt,3)

    # discretize velocity space
    wmax = u.convert(1000.*u.km/u.s, u.dimensionless_unscaled, value=True)    
    N = 100
    
    wU_grid = np.linspace(-wmax, wmax, N) 
    wV_grid = np.linspace(-wmax, wmax, N)
    wW_grid = np.linspace(-wmax, wmax, N)
    
    dU = np.diff(wU_grid)[0]
    dV = np.diff(wV_grid)[0]
    dW = np.diff(wW_grid)[0]

    wU, wV, wW = np.meshgrid(wU_grid, wV_grid, wW_grid, indexing='ij') # (N,N,N)
    
    wvec = np.stack((wU, wV, wW),axis=-1) # (N, N, N, 3)
    fw = f(wvec) # (N, N, N)

    # iterate over time
    output = np.zeros(nt)
    if verbose:
        iterator = tqdm(range(nt), desc="Computing Two-Point Correlation Function")
    else:
        iterator = range(nt)
    for i in iterator:
        # build integrate
        v1 = wvec + vobs_1[i] # (N, N, N, 3)
        v2 = wvec + vobs_2[i] # (N, N, N, 3)
        vi = v1 @ ihat # (N, N, N) dot product
        vj = v2 @ jhat # (N, N, N) dot product
        v1_squared = np.sum(v1**2, axis=-1) # (N, N, N)
        v2_squared = np.sum(v2**2, axis=-1) # (N, N, N)
        omega1 = 0.5*v1_squared*(m*t1_flat[i]).to_value(u.dimensionless_unscaled) # (N, N, N)
        omega2 = 0.5*v2_squared*(m*t2_flat[i]).to_value(u.dimensionless_unscaled) # (N, N, N)
        integrand = vi*vj*fw*trig(omega2-omega1) # (N, N, N)

        # integrate using 3-dimensional trapezoidal rule
        output[i] = (dU*dV*dW/8.)*np.sum(integrand[1:,1:,1:] + integrand[1:,1:,:-1] + integrand[1:,:-1,1:] + integrand[1:,:-1,:-1] + integrand[:-1,1:,1:] + integrand[:-1,1:,:-1] + integrand[:-1,:-1,1:] + integrand[:-1,:-1,:-1]) # scalar

    # make sure to return output with correct shape
    return output.reshape(t_shape)
    
