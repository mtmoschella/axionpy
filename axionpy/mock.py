"""
This module for generating mock data.
"""
import scipy.interpolate as interp
import numpy as np
from tqdm import tqdm

from axionpy import maxwell
from axionpy import axis as axs

def signal(m, g, **kwargs):
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

    dt : astropy.Quantity (time)
         The time step of time series.
         Either t or N and dt must be specified.

    components : (3,N) array-like
                 The components of the sensitive axis in the xyz basis
                 evaluated over the time series t.
                 One of components, axis, or (lat, lon, theta, phi) must be specified.    

    axis : Axis 
           Axis object representing the sensitive axis
           One of components, axis, or (lat, lon, theta, phi) must be specified.

    lat, lon, theta, phi : float
           The latitude, longitude, and orientation of the sensitive axis.
           One of components, axis, or (lat, lon, theta, phi) must be specified.

    ntrials : (optional) int
              If specified, the number of independent signals to generate.
              If specified, the output will be shape (N, ntrials), otherwise
              the output will be shape (N,)

    epoch : (optional) str or astropy.Time
            Epoch specifying the starting time. Defaults to J2000. 
            All times are measured from this epoch.

    fname : (optional) str
            Filename or path to file.
            If specified creates (or overwrites) a numpy.memmap array for the output
            at the specified location. Returns a read-only numpy.memmap.

    deltat : (optional) astropy.Quantity (time)
             The size of the discretized bins used to compute the stochastic evolution.
             The time evolution of the stochastic coefficients is interpolated on timescales
             smaller than deltat.
             Defaults to 0.1*tau (coherence time).
             Raises warning if deltat >= tau

             If deltat is large enough that fewer than 5 bins are needed, it is disregarded
             and 5 bins are used.

    buffersize : (optional) int
                 The maximum size of an array to read into memory at one time.
                 Defaults to 10,000,000
                 Note: multiple arrays of size (buffersize,) may in memory at once.

    verbose : (optional) boolean
              Display progress bar on screen.
              Defaults to False.
    Returns
    ---------------------
    signal : (N,) or (ntrials, N) astropy.Quantity (GeV) or numpy.memmap
             The gradient of the axion field over the specified time series.
             If fname is specified, the output is a memmap ndarray in units of GeV.
    """
    # parse kwargs
    if 'buffersize' in kwargs:
        buffersize = kwargs['buffersize']
    else:
        buffersize = int(1.e7)
        
    if "N" in kwargs and "dt" in kwargs:
        N = int(kwargs["N"])
        dt = kwargs["dt"]
        tmax = N*dt
    elif "t" in kwargs:
        t = kwargs["t"]
        N = len(t)
        if N>buffersize:
            print("WARNING: The specified time array is larger than the buffersize.")
        dt = None
        tmax = np.amax(t)-np.amin(t)
    else:
        raise Exception("ERROR: Invalid kwargs. Must specify either N and dt or t.")

    nbuffers = np.ceil(N/buffersize).astype(int)
    
    if 'components' in kwargs:
        components = kwargs['components']
    else:
        if 'axis' in kwargs:
            axis = kwargs['axis']
        elif 'lat' in kwargs and 'lon' in kwargs and 'theta' in kwargs and 'phi' in kwargs:
            axis = axs.Axis(kwargs['lat'], kwargs['lon'], kwargs['theta'], kwargs['phi'])
        else:
            raise Exception("ERROR: must specify components, axis or (lat, lon, theta, phi)")

        axis_kwargs = {}
        if 'epoch' in kwargs:
            axis_kwargs['epoch'] = kwargs['epoch']
        components = axis.components(t=t, basis='xyz', **axis_kwargs)
    
    if 'ntrials' in kwargs:
        ntrials = int(kwargs['ntrials'])
        output_shape = (ntrials, N)
    else:
        ntrials = 1
        output_shape = (N,)
        
    if 'fname' in kwargs:
        use_memmap = True
        output = np.memmap(kwargs['fname'], dtype=float, mode='w+', shape=output_shape)
        geff = u.convert(np.sqrt(_rhodm)*g, u.GeV, value=True) # convert from dimensionlss to output units
    else:
        use_memmap = False
        output = np.empty(output_shape)*u.GeV
        geff = u.convert(np.sqrt(_rhodm)*g, u.GeV) # convert from dimensionlss to output units

    if 'verbose'in kwargs:
        verbose = bool(kwargs['verbose'])
    else:
        verbose = False
        
    ##### discretize time series

    # parse bin size
    tau = 1./(m*maxwell._sigma**2) # axion coherence time
    if 'deltat' in kwargs:
        deltat = kwargs['deltat']
        if deltat >= tau:
            print("WARNING: The specified deltat is too large compared to the coherence time")
    else:
        deltat = 0.1*tau

    # minimum of 5 bins
    if deltat > 0.2*tmax:
        deltat = 0.2*tmax

    t_discrete = np.arange(0.0, tmax.to_value(u.day), deltat.to_value(u.day))*u.day # (M,)
    M = len(t_discrete) 

    ###### generate random coefficients
    t1, t2 = np.meshgrid(t_discrete, t_discrete, indexing='ij') # (M,M)

    # dimensionless two-point correlators
    AxAx = maxwell.correlator('AxAx', m, t1, t2, **kwargs) # (M,M)
    AxBx = maxwell.correlator('AxBx', m, t1, t2, **kwargs)
    AyAy = maxwell.correlator('AyAy', m, t1, t2, **kwargs)
    AyBy = maxwell.correlator('AyBy', m, t1, t2, **kwargs)
    AzAz = maxwell.correlator('AzAz', m, t1, t2, **kwargs)
    AzBz = maxwell.correlator('AzBz', m, t1, t2, **kwargs)

    # dimensionless covariance matries
    Cx = np.block([[AxAx, AxBx], [-AxBx, AxAx]]) # (2M,2M)
    Cy = np.block([[AyAy, AyBy], [-AyBy, AyAy]])
    Cz = np.block([[AzAz, AzBz], [-AzBz, AzAz]])

    # all RVs have mean zero
    mu = np.zeros(2*M) 

    # generate dimensionless gaussian RVs 
    Xx = np.random.multivariate_normal(mu, Cx, size=ntrials) # (ntrials, 2M)
    Xy = np.random.multivariate_normal(mu, Cy, size=ntrials) 
    Xz = np.random.multivariate_normal(mu, Cz, size=ntrials)

    # separate dimensionless A and B coefficients
    Ax_discrete = Xx[...,:M]
    Bx_discrete = Xx[...,M:]
    Ay_discrete = Xy[...,:M]
    By_discrete = Xy[...,M:]
    Az_discrete = Xz[...,:M]
    Bz_discrete = Xz[...,M:]
    
    ###### generate full time series
    
    # prepare for interpolation
    t_interp = t_discrete.to_value(u.day) # interpolation must be dimensionless values

    # prepare iterators (with progress bar if verbose is specified)
    trials_iterator = range(ntrials)
    buffers_iterator = range(nbuffers)
    if verbose:
        if ntrials>1:
            # show progress bar for iteration over trials
            trials_iterator = tqdm(trials_iterator)
        else:
            # who progress bar for the buffers within a single trial
            buffers_iterator = tqdm(buffers_iterator)

    # iterate over random trials
    for trial in trials_iterator:
        # interpolate the coefficients
        Ax_tck = interp.splrep(t_interp, Ax_discrete[trial])
        Bx_tck = interp.splrep(t_interp, Bx_discrete[trial])
        Ay_tck = interp.splrep(t_interp, Ay_discrete[trial])
        By_tck = interp.splrep(t_interp, By_discrete[trial])
        Az_tck = interp.splrep(t_interp, Az_discrete[trial])
        Bz_tck = interp.splrep(t_interp, Bz_discrete[trial])

        # iterate over memory buffers
        for buff in buffers_iterator:
            start = buff*buffersize
            stop = min(start+buffersize, N)
            if start>=stop:
                # redundant failsafe
                break

            # generate time series if necessary
            if dt is None:
                t_buffer = t[start:stop]
            else:
                t_buffer = np.arange(start, stop)*dt

            t_eval = t_buffer.to_value(u.day) # for evaluating dimensionless interpolation

            # evaluate interpolated coefficients
            Ax = interp.splev(t_eval, Ax_tk)
            Bx = interp.splev(t_eval, Bx_tk)
            Ay = interp.splev(t_eval, Ay_tk)
            By = interp.splev(t_eval, By_tk)
            Az = interp.splev(t_eval, Az_tk)
            Bz = interp.splev(t_eval, Bz_tk)

            # generate full-frequency time resolution
            mt = (m*t_buffer).to_value(u.dimensionless_unscaled)
            cos = np.cos(mt)
            sin = np.sin(mt)
            s = (Ax*cos+Bx*sin)*components[0,start:stop] + (Ay*cos+By*sin)*components[1,start:stop] + (Az*cos+Bz*sin)*components[2,start:stop] # dimensionles

            if len(output_shape)==1:
                output[start:stop] = geff*s # convert to appropriate output unit
            else:
                output[trial,start:stop] = geff*s # convert to appropriate output unit
            
    if use_memmap:
        # if using a memmap, delete the w+ memmap and return a read-only memmap
        del output
        output = np.memmap(fname, dtype=float, mode='r', shape=output_shape)
        
    return output
    
            
