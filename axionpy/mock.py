"""
This module for generating mock data.
"""

def signal(m, g, t, **kwargs):
    """
    Generates a random signal over the specified time series.

    Parameters
    ------------------
    m : astropy.Quantity (frequency)
        The axion mass.

    g : astropy.Quantity (GeV^-1)
        The axion coupling constant.

    t : (N,) astropy.Quantity (time)
        The time series over which to generate the signal.
        Either t or N and dt must be specified.

    N : int 
        The size of time series.
        Either t or N and dt must be specified.    

    dt : astropy.Quantity
         The time step of time series.
         Either t or N and dt must be specified.
n
    ntrials : int
              If specified, the number of independent signals to generate.
              If specified, the output will be shape (N, ntrials), otherwise
              the output will be shape (N,)

    components : (3,N) array-like
                 The components of the sensitive axis evaluated over the time series t.
                 One of components, axis, or (lat, lon, theta, phi) must be specified.    

    axis : Axis 
           Axis object representing the sensitive axis
           One of components, axis, or (lat, lon, theta, phi) must be specified.

    lat, lon, theta, phi : float
           The latitude, longitude, and orientation of the sensitive axis.
           One of components, axis, or (lat, lon, theta, phi) must be specified.

    epoch : str or astropy.Time
            Epoch specifying the starting time. Defaults to J2000. 
            All times are measured from this epoch.

    fname : str
            Filename or path to file.
            If specified creates (or overwrites) a numpy.memmap array for the output
            at the specified location.

    Returns
    ---------------------
    signal : (N,) or (N, ntrials) astropy.Quantity (GeV)
             The gradient of the axion field over the specified time series.
    """
    pass




