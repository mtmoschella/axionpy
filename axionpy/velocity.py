import numpy as np
import astropy.coordinates as coord
import axionpy.units as u
# all vectors in Galactic (u,v,w) basis unless otherwise specified
# u: towards the galactic center
# v: direction of disk rotation (East)
# w: normal to the disk (North)

_u = np.array([1., 0., 0.])
_v = np.array([0., 1., 0.])
_w = np.array([0., 0., 1.])

vcirc = 220.
vlsr = np.array([0., vcirc, 0.]) # km/s
vpec = np.array([11., 12., 7.])
vsun = vlsr + vpec

gc_coord = coord.ICRS(ra=266.4051*u.deg, dec=-28.936175*u.deg)
gc_distance = 8.3*u.kpc
z_sun = 27.*u.pc
roll = 0.0*u.deg

def _vobs(loc, t):
    """
    Parameters
    ------------
    loc : astropy.EarthLocation
    t : astropy.Time

    Returns:
    ------------
    v : astropy.Quantity of shape np.shape(t)+(3,)
        The velocity of the specified EarthLocation in Galactocentric (u,v,w) coordinates.
    """
    gcrs = loc.get_gcrs(t)
    galactocentric = gcrs.transform_to(coord.Galactocentric(galcen_coord=gc_coord, galcen_distance=gc_distance, galcen_v_sun=vsun*u.km/u.s, z_sun=z_sun, roll=roll))
    galactocentric.representation_type = 'cartesian'
    # (x,y,z) here refers to the astropy.Galactocentric basis, which I call (u,v,w)
    vu = galactocentric.v_x.to_value(u.km/u.s)
    vv = galactocentric.v_y.to_value(u.km/u.s)
    vw = galactocentric.v_z.to_value(u.km/u.s)
    return np.stack((vu, vv, vw), axis=-1)*u.km/u.s
    

def _unit(arr):
    """
    arr: numpy ndarray
    
    Given ndarray of shape (n1, ..., nN)
    Renormalize such that array can be interpreted 
    as an (n2,...,nN) array of n1-dimensional unit vectors
    """
    return arr/np.sqrt(np.sum(arr**2, axis=0))

###### default velocity basis
# choose _zhat parallel to vsun
# choose _xhat perpendicular to galactic North (out of the plane)
# choose _yhat to complete RH coordinate system: _xhat x _yhat = _zhat
#              _yhat = _zhat  _xhat
_z = _unit(vsun)
_x = np.cross(_z, _w)
_y = np.cross(_z, _x)

