import numpy as np

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
