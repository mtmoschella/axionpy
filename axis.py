import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from velocity import vsun, _u, _v, _w
import scipy.interpolate as interp
import util_toolkit as util
import os
from tqdm import tqdm

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
        
    def components(self, N, dt, start='2020-01-01T00:00:00', xhat=_x, yhat=_y, zhat=_z, tstep_min=0.5*u.hr, buffersize=int(1.e7), fname='components.dat', overwrite=False, pbar=True):
        """
        N: size of time series
        dt: astropy.quantity (time)
        start: date/time string or astropy Time
        xhat, yhat, zhat: basis to get components

        Returns a memmap of shape (3,N) such that
        output[0,:] = x component
        output[1,:] = y component
        output[2,:] = z component
        """

        if overwrite or not os.path.exists(fname):
            memmap = np.memmap(fname, dtype=float, mode='w+', shape=(3,N))
            
            tstart = Time(start)
            
            if dt>tstep_min:
                # don't interpolate
                i=0
                if pbar: bar=tqdm(total=int(np.ceil(N/buffersize)))
                while i<N:
                    n = min(buffersize, N-i)
                    dirs = self._time_series(tstart+i*dt, n, dt) # (3, n)
                    memmap[0,i:i+n] = np.einsum('ij,i->j', dirs, xhat) # (n,)
                    memmap[1,i:i+n] = np.einsum('ij,i->j', dirs, yhat) # (n,)
                    memmap[2,i:i+n] = np.einsum('ij,i->j', dirs, zhat) # (n,)
                    i += n
                    if pbar: bar.update(1)
                if pbar: bar.close()
            else:
                # interpolate
                M = int(1+np.ceil((N*dt/tstep_min)))
                if M>buffersize:
                    raise Exception("ERROR: interpolation beyond buffersize is not supported")
                else:
                    # M is small enough to read into memory directly
                    tguess = (1./49.)*M # guess how long to compute interpolation points
                    if tguess>5.:
                        print("Computing "+str(M)+" interpolation points. Should take less than "+str(tguess)+" s")
                    dirs = self._time_series(tstart, M, tstep_min) # (3, M)
                    x = np.einsum('ij,i->j', dirs, xhat) # (M,)
                    y = np.einsum('ij,i->j', dirs, yhat) # (M,)
                    z = np.einsum('ij,i->j', dirs, zhat) # (M,)

                    tstepped = np.arange(M)*tstep_min.to_value(u.s) # (M,)
                    x_tck = interp.splrep(tstepped, x)
                    y_tck = interp.splrep(tstepped, y)
                    z_tck = interp.splrep(tstepped, z)

                    i=0
                    if pbar: bar=tqdm(total=int(np.ceil(N/buffersize)))
                    while i<N:
                        n = min(buffersize, N-i)
                        t = np.arange(i,i+n)*dt.to_value(u.s) # (N,)
                        memmap[0,i:i+n] = interp.splev(t, x_tck) # (N,)
                        memmap[1,i:i+n] = interp.splev(t, y_tck) # (N,)
                        memmap[2,i:i+n] = interp.splev(t, z_tck) # (N,)
                        i += n
                        if pbar: bar.update(1)
                    if pbar: bar.close()
            del memmap
        if fname=='components.dat':
            print("WARNING: Reading default memmap file components.dat")
        output = np.memmap(fname, dtype=float, mode='r', shape=(3,N))
        return output
    
