import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import util_toolkit as util
import os
from tqdm import tqdm

from .velocity import vsun, _u, _v, _w

def _is_quantity(q, t=None):
    """
    Verifies if the argument is an astropy.Quantity of the specified type
    
    q: astropy.Quantity (expected)
       Argument to check type

    t: string, optional
       If specified, additionally requires the physical_type of the quantity to match t
    
    Returns: bool
             True if the specified quantity matches.
             False if not.
    """
    if not isinstance(q, u.Quantity):
        return False
    if t is not None:
        return q.unit.physical_type==t
    return True

def _unit(arr):
    """
    arr: numpy ndarray
    axis: axis along which to renormalize
    
    Given ndarray of shape (n1, ..., nN)
    Renormalize such that array can be interpreted 
    as an (n2,...,nN) array of n1-dimensional unit vectors
    """
    return arr/np.sqrt(np.sum(arr**2, axis=0))

# Earth-based coordinate basis
_oEarth = coord.EarthLocation(x=0.*u.m, y=0.*u.m, z=0.*u.m)
_xEarth = coord.EarthLocation(lon=0., lat=0., height=0.)
_yEarth = coord.EarthLocation(lon=90., lat=0., height=0.)
_zEarth = coord.EarthLocation(lon=0., lat=90., height=0.)

###### default velocity basis
# choose _zhat parallel to vsun
# choose _xhat perpendicular to galactic North (out of the plane)
# choose _yhat to complete RH coordinate system: _xhat x _yhat = _zhat
#              _yhat = _zhat  _xhat
_z = _unit(vsun)
_x = np.cross(_z, _w)
_y = np.cross(_z, _x)


def _EarthLocation_to_Galactic(loc, t):
    """
    loc: astropy EarthLocation 
    t: astropy Time quantity

    Returns the Galactic coordinates (u,v,w) as astropy.quantity
    """
    gcrs = loc.get_gcrs(t)
    galactic = gcrs.transform_to(coord.Galactic())
    galactic.representation_type = 'cartesian'
    return galactic.u, galactic.v, galactic.w

def _EarthLocation_to_Galactic_vector(loc, t):
    """
    loc: astropy EarthLocation
    t: astropy Time
    
    Returns a vector of shape (3,) + t.shape
    corresponding to the vector from the center of the Earth
    to the specified Earth location, in Galactic (u,v,w) coordinates
    as an astropy.quantity (distance)
    """
    loc_u, loc_v, loc_w = _EarthLocation_to_Galactic(loc, t)
    o_u, o_v, o_w = _EarthLocation_to_Galactic(_oEarth, t)
    return np.array([ (loc_u-o_u).to_value(u.m), (loc_v-o_v).to_value(u.m), (loc_w-o_w).to_value(u.m)])*u.m

def _Earth_basis_vectors(t):
    """
    t: astropy Time quantity

    Returns the Earth coordinate basis vectors xhat, yhat, zhat
    in Galactic coordinates (u,v,w)

    xhat: direction from center of Earth to 0 lat, 0 lon
    yhat: direction from center of Earth to 0 lat, 90 lon
    zhat: direction from center of Earth to North Pole (90 lat, undefined lon)
    """

    xEarth_vec = _EarthLocation_to_Galactic_vector(_xEarth, t).to_value(u.m)
    yEarth_vec = _EarthLocation_to_Galactic_vector(_yEarth, t).to_value(u.m)
    zEarth_vec = _EarthLocation_to_Galactic_vector(_zEarth, t).to_value(u.m)
    
    # get unit vectors
    xhat = xEarth_vec/np.sqrt(np.sum(xEarth_vec**2, axis=0))
    yhat = yEarth_vec/np.sqrt(np.sum(yEarth_vec**2, axis=0))
    zhat = zEarth_vec/np.sqrt(np.sum(zEarth_vec**2, axis=0))

    return xhat, yhat, zhat

def _local_basis_vectors(lat, lon, xhat, yhat, zhat):
    """
    lat, lon: latitude and longitude
    xhat, yhat, zhat: Earth units basis vectors

    Returns the local basis vectors east, north, up at the specified lat and lon
    """
    up = np.cos(lat)*np.cos(lon)*xhat + np.cos(lat)*np.sin(lon)*yhat + np.sin(lat)*zhat
    east = -np.sin(lon)*xhat + np.cos(lon)*yhat
    north = -np.sin(lat)*np.cos(lon)*xhat - np.sin(lat)*np.sin(lon)*yhat + np.cos(lat)*zhat
    return east, north, up

class Axis:
    """
    Represents a direction on the surface of the Earth.
    """
    def __init__(self, lat, lon, theta, phi):
        """
        lat, lon: latitude and longitude of the location of the axis on the Earth
        theta: declination of the axis (polar angle measured from azimuth)
        phi: orientation of the axis (azimuthal angle measured CCW from East)
        """
        self.lat = lat
        self.lon = lon
        self.theta = theta
        self.phi = phi

    def _dir(self, t):
        """
        t: astropy Time

        Returns the direction in Galactic coordinates
        """
        # get Earth basis
        xhat, yhat, zhat = _Earth_basis_vectors(t)

        # get local basis
        east, north, up = _local_basis_vectors(self.lat, self.lon, xhat, yhat, zhat)

        direction = np.sin(self.theta)*np.cos(self.phi)*east + np.sin(self.theta)*np.sin(self.phi)*north + np.cos(self.theta)*up
        return direction

    def _time_series(self, tstart, M, tstep=100.*u.s):
        """
        tstart: astropy Time
        M: integer
        tstep: astropy.quantity (time)

        Returns (3, M) array
        """
        t = tstart + np.arange(M)*tstep
        return self._dir(t)

    #def components(self, N=None, dt=None, tgrid=None, start='2020-01-01T00:00:00', xhat=_x, yhat=_y, zhat=_z, tstep_min=0.5*u.hr, buffersize=int(1.e7), fname='components.dat', overwrite=False, pbar=True):    
    def components(self, **kwargs):
        """
        ===========================
        Required kwargs: 
        Either N and dt or t must be specified.
        --------------------------
        N : int 
            Size of time series

        dt : astropy.Quantity
             Time step of time series

        t : (N,) astropy.Quantity 
            Full time series 

        ==========================
        optional kwargs
        --------------------------
        epoch: str or astropy.Time
               Epoch specifying the starting time. Defaults to J2000. 
               All times are measured from this epoch.

        xhat : (3,) array-like
               x basis vector in galactic {u,v,w} coordinates.
               Defaults to _x.

        yhat : (3,) array-like
               y basis vector in galactic {u,v,w} coordinates.
               Defaults to _y.

        zhat : (3,) array-like
               z basis vector in galactic {u,v,w} coordinates. 
               Defaults to _z.

        tstep_min : astropy.Quantity
                    Minimum time step for using full astropy EarthLocation evaluation. 
                    Time steps shorter than this will be evaluated using interpolation.
                    Defaults to 30 minutes.

        ================================
        Output
        --------------------------------
        Returns a numpy.array of shape (3,N) such that
        output[0,:] = x component (N,)
        output[1,:] = y component (N,)
        output[2,:] = z component (N,)
        """

        # parse required kwargs
        if "N" in kwargs and "dt" in kwargs:
            N = int(kwargs["N"])
            dt = kwargs["dt"]
            if not _is_quantity(dt, "time"):
                raise Exception("ERROR: dt is required to be an astropy.Quantity of time")
        elif "t" in kwargs:
            t = kwargs["t"]
            N = len(t)
            dt = None
            if not _isquantity(t, "time"):
                raise Exception("ERROR: t is required to be an astropy.Quantity of time")
        else:
            raise Exception("ERROR: Invalid kwargs. Must specify either N and dt or t.")

        # parse optional kwargs
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
            if not isinstance(epoch, Time):
                epoch = Time(epoch)
        else:
             epoch = Time("J2000.0")

        if 'xhat' in kwargs:
            xhat = np.asarray(kwargs['xhat'])
            if xhat.shape!=(3,):
                raise Exception("ERROR: xhat must have shape (3,)")
        else:
            xhat = _x

        if 'yhat' in kwargs:
            yhat = np.asarray(kwargs['yhat'])
            if yhat.shape!=(3,):
                raise Exception("ERROR: yhat must have shape (3,)")
        else:
            yhat = _y

        if 'zhat' in kwargs:
            zhat = np.asarray(kwargs['zhat'])
            if zhat.shape!=(3,):
                raise Exception("ERROR: zhat must have shape (3,)")            
        else:
            zhat = _zhat

        if 'tstep_min' in kwargs:
            tstep_min = kwargs['tstep_min']
            if not _is_quantity(tstep_min, "time"):
                raise Exception("ERROR: tstep_min is required to be an astropy.Quantity of time")
        else:
            tstep_min = 30.*u.min

        output = np.empty((3,N))

        # decide whether or not to interpolate
        if dt is None:
            # full time series specified
            deltat = np.amax(t)-np.amin(t)
        else:
            # N, dt specified
            deltat = N*dt
        use_interp = N*tstep_min > deltat
            
        if not use_interp:
            # don't interpolate
            # just evaluate time series directly
            if dt is None:
                dirs = self._dir(epoch+t) # (3, N)
            else:
                dirs = self._time_series(epoch, N, dt) # (3, N)
            output[0,:] = np.einsum('ij,i->j', dirs, xhat) # (N,)
            output[1,:] = np.einsum('ij,i->j', dirs, yhat) # (N,)
            output[2,:] = np.einsum('ij,i->j', dirs, zhat) # (N,)
        else:
            # use interpolation
            M = int(1+np.ceil((deltat/tstep_min)))
            
            dirs = self._time_series(epoch, M, tstep_min) # (3, M)
            x = np.einsum('ij,i->j', dirs, xhat) # (M,)
            y = np.einsum('ij,i->j', dirs, yhat) # (M,)
            z = np.einsum('ij,i->j', dirs, zhat) # (M,)

            tstepped = np.arange(M)*tstep_min.to_value(u.s) # (M,)
            x_tck = interp.splrep(tstepped, x)
            y_tck = interp.splrep(tstepped, y)
            z_tck = interp.splrep(tstepped, z)

            if dt is None:
                t_eval = epoch+t
            else:
                t_eval = np.arange(N)*dt.to_value(u.s) # (N,)
            output[0,:] = interp.splev(t_eval, x_tck) # (N,)
            output[1,:] = interp.splev(t_eval, y_tck) # (N,)
            output[2,:] = interp.splev(t_eval, z_tck) # (N,)
        return output
    
