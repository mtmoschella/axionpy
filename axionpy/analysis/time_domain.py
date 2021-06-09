"""
This module for the "full" time domain analysis.

This analysis procedure works by splitting the time series into bins
that are much smaller than the axion coherence time and much smaller than 1 day.
Therefore, the axion oscillations can be treated as coherent and having constant amplitude
over the course of each bin.
The amplitude and phase of the coherent axion oscillations in each bin is obtained via 
ordinary least-squares fitting and then the time series of these coefficients is input into 
a Gaussian likelihood with non-trivial covariance matrix computed by the cov() function.

This procedure is only valid when the axion mass satisfies m >~ 0.1*2*pi/day, 
otherwise the axion does not oscillate rapidly enough to uniquely identify the
amplitude and phase of oscilliations in each time bin.
"""
import numpy as np 
import scipy.optimize as opt
from scipy import stats
from axionpy import units as u
from axionpy.distributions import maxwell
from axionpy import axis as ax
from axionpy.constants import _rhodm
from axionpy import matrix

def cov(m, t, **kwargs):
    """
    m : astropy.Quantity
    t : (N,) astropy.Quantity

    components : (optional) (3,N) array-like
                 The components of the sensitive axis in the xyz basis
                 evaluated over the time series t.
                 One of components, axis, or (lat, lon, theta, phi) must be specified.    

    axis : (optional) Axis 
           Axis object representing the sensitive axis
           One of components, axis, or (lat, lon, theta, phi) must be specified.

    lat, lon, theta, phi : (optional) float
           The latitude, longitude, and orientation of the sensitive axis.
           One of components, axis, or (lat, lon, theta, phi) must be specified.
    
    kwargs : passed to Axis.components if components is not specified
    
    Return CovMatrix object
    """
    if 'components' in kwargs:
        components = kwargs['components']
    else:
        if 'axis' in kwargs:
            axis = kwargs['axis']
        elif 'lat' in kwargs and 'lon' in kwargs and 'theta' in kwargs and 'phi' in kwargs:
            axis = ax.Axis(kwargs['lat'], kwargs['lon'], kwargs['theta'], kwargs['phi'])
        else:
            raise Exception("ERROR: must specify components, axis or (lat, lon, theta, phi)")

        axis_kwargs = {}
        if 'epoch' in kwargs:
            axis_kwargs['epoch'] = kwargs['epoch']
        if 'tstep_min' in kwargs:
            axis_kwargs['tstep_min'] = kwargs['tstep_min']
        if 'verbose' in kwargs:
            axis_kwargs['verbose'] = kwargs['verbose']
        components = axis.components(t=t, basis='xyz', **axis_kwargs)
    
    t1, t2 = np.meshgrid(t, t, indexing='ij')
    mx_1, mx_2 = np.meshgrid(components[0], components[0], indexing='ij')
    my_1, my_2 = np.meshgrid(components[1], components[1], indexing='ij')
    mz_1, mz_2 = np.meshgrid(components[2], components[2], indexing='ij')

    AxAx = maxwell.correlator('AxAx', m, t1, t2, **kwargs)
    AxBx = maxwell.correlator('AxBx', m, t1, t2, **kwargs)
    AyAy = maxwell.correlator('AyAy', m, t1, t2, **kwargs)
    AyBy = maxwell.correlator('AyBy', m, t1, t2, **kwargs)
    AzAz = maxwell.correlator('AzAz', m, t1, t2, **kwargs)
    AzBz = maxwell.correlator('AzBz', m, t1, t2, **kwargs)

    AA = mx_1*mx_2*AxAx + my_1*my_2*AyAy + mz_1*mz_2*AzAz
    AB = mx_1*mx_2*AxBx + my_1*my_2*AyBy + mz_1*mz_2*AzBz

    C = np.block([[AA, AB], [-AB, AA]])

    return matrix.CovMatrix(C)

def compute_coefficients(m, t, y, err=False):
    """
    Compute the A and B coefficients for the given time series of data.

    These are defines such that
    S(t) \propto A*cos(m*t) - B*sin(m*t)
    (Note the sign convention for the B coefficient)
    
    Physically, this only makes sense if the total time interval
    is much shorter than both 1 day and the axion coherence time.

    Parameters
    ---------------
    m : astropy.Quantity (frequency)
        The axion mass. 
        NOTE: This is treated as an angular frequency.

    t : (N,) astropy.Quantity (time)
        The time series.

    y : (N,) astropy.Quantity
        The observed data. Can have arbitrary astropy units.

    err : bool
          If true, returns the Gaussian error estimate for A and B.
          This is equal to sqrt(2*SSR)/N.

    Returns
    ---------------
    A, B : astropy.Quantity
           The A and B coefficients such that 
           y = A*cos(m*t) + B*sin(m*t) is the least-squares solution
           Output is an astropy.Quantity with the same units as y

    sigma : astropy.Quantity
            Only returned if err == True
            The error estimate for A and B in the same units as y
    """
    # check if valid time series
    T = np.amax(t)-np.amin(t)
    if T>1.e5/m or T>0.1*u.day:
        print("WARNING: compute_coefficients only makes physical sense on short timescales.")

    unit = y.unit
    yval = y.to_value(unit)
    mt = (m*t).to_value(u.dimensionless_unscaled) # radians
    
    M = np.transpose([ np.cos(mt), -1.*np.sin(mt) ]) # (N, 2)
    coeffs, resid, rank, s = np.linalg.lstsq(M, yval, rcond=None)
    A, B = coeffs
    if err:
        sigma = np.sqrt(2.*resid[0])/len(yval)
        return A*unit, B*unit, sigma*unit
    else:
        return A*unit, B*unit
        
def loglikelihood(x, g, s, c):
    """
    Computes the loglikelihood of the data given model parameters.

    Parameters
    ----------------
    x : (2N,) astropy.Quantity (GeV)
        The A,B fit coefficients in each bin of the time series.
        Stacked such that x  = (A1, ..., AN, B1, ..., BN)

    g : astropy.Quantity (GeV^-1 or natural equivalent)
        The axion coupling constant.

    s : astropy.Quantity (GeV)
        The constant uncertainty on x, the binned coefficients.
    
    c : CovMatrix
        Object representing the (2N,2N) covariance matrix

    Returns
    ------------------
    ll : float
         The log-likelihood evaluated with the given data and model parameters.
    """
    geff = u.convert(np.sqrt(_rhodm)*g, u.GeV) # rescale for covariance matrix
    chi2 = c.chi2(geff.to_value(u.GeV), s.to_value(u.GeV), x.to_value(u.GeV))
    logdet = c.logdet(geff.to_value(u.GeV), s.to_value(u.GeV))
    return -0.5*(chi2 + logdet)

def maximize_likelihood(A, B, c, g_scale=1.0e-10*u.GeV**-1, s_scale=1.e-34*u.GeV):
    """
    Maximizes the likelihood with respect to the axion coupling g and the noise parameter s.

    Parameters:
    ------------------------
    A,B : (N,) astropy.Quantity (GeV)
          The fit coefficients in each bin of the time series.

    c : CovMatrix
        Object representing the (2N, 2N) covariance matrix

    g_scale : (optional) astropy.Quantity (GeV^-1 or natural equivalent)
              Rough scale/estimate for the coupling g.
              Used for initial conditions in the optimization routine.
              Defaults to 1.e-10*u.GeV**-1

    s_scale: (optional) astropy.Quantity (GeV)
             Rough scale/estimate for the noise parameter s.
             Used for initial conditions in the optimization routine.             
             Defaults to 1.e-34*u.GeV

    Returns:
    ------------------------
    g : astropy.Quantity (GeV^-1)
        The best-fit coupling constant.

    s : astropy.Quantity (GeV)
        The best-fit noise parameter.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """
    x = np.concatenate((A,B))
    
    def f_to_minimize(p):
        log10g, log10s = p
        ll = loglikelihood(x, 10.**log10g*u.GeV**-1, 10.**log10s*u.GeV, c)
        if np.isfinite(ll):
            return -1.*ll
        raise Exception("ERROR: non-finite ll")

    p0 = np.array([ np.log10(g_scale.to_value(u.GeV**-1)), np.log10(s_scale.to_value(u.GeV)) ])
    res = opt.minimize(f_to_minimize, p0, bounds=[[None, 0.0], [None, 0.0]])
    log10g, log10s = res.x
    maxll = -1.*res.fun
    return 10.**log10g*u.GeV**-1, 10.**log10s*u.GeV, maxll

def profile_likelihood(A, B, c, g, s_scale=1.e-34*u.GeV):
    """
    Maximizes the likelihood with respect to the noise parameter s for a given value of axion coupling g.

    Parameters:
    ------------------
    A,B : (N,) astropy.Quantity (GeV)
          The fit coefficients in each bin of the time series.

    c : CovMatrix
        Object representing the (2N, 2N) covariance matrix

    g : astropy.Quantity (GeV^-1)
        The fixed value of the axion coupling.

    s_scale: (optional) astropy.Quantity (GeV)
             Rough scale/estimate for the noise parameter s.
             Used for initial conditions in the optimization routine.             
             Defaults to 1.e-34*u.GeV

    Returns:
    ------------------------
    s : astropy.Quantity (GeV)
        The best-fit noise parameter.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """
    x = np.concatenate((A,B))

    def f_to_minimize(p):
        log10s = p
        ll = loglikelihood(x, g, 10.**log10s*u.GeV, c)
        if np.isfinite(ll):
            return -1.*ll
        raise Exception("ERROR: non-finite ll")

    p0 = np.log10(s_scale.to_value(u.GeV))
    res = opt.minimize(f_to_minimize, p0)
    log10s = res.x
    maxll = -1.*res.fun
    return 10.**log10s*u.GeV, maxll

def frequentist_upper_limit(A, B, c, confidence=0.95, gmax=None, smax=None, llmax=None, g_scale=1.e-10*u.GeV**-1, s_scale=1.e-34*u.GeV):
    """
    Computes the axion coupling constant g that is the frequentist upper limit 
    at the specified constant, that is, the value of g such that the profile likelihood ll 
    satisfies chi2 = 2*(llmax-ll), 
    where chi2 is the value of the chi-squared distribution with one degree of freedom 
    such that cdf(chi2) = 1 - 2*(1-confidence). 

    Note that the reason this does not satisfy cdf(chi2)=confidence 
    is because Wilks' theorem guarantees that the test statistic chi2 
    is a half-chi-squared distributed random variable under the null hypothesis, 
    rather than a full-chi-squared distriubed random variable.

    Parameters:
    ------------------------
    A,B : (N,) astropy.Quantity (GeV)
          The fit coefficients in each bin of the time series.

    c : CovMatrix
        Object representing the (2N, 2N) covariance matrix
    
    confidence : (optional) float
                 The requested confidence for the upper limit. 
                 Must satify 0.<confidence<1.
                 Defaults to 0.95

    gmax : (optional) astropy.Quantity (GeV^-1)
           The best-fit coupling constant. 
           If not specified then the maximize_likelihood function is called.
    
    smax : (optional) astropy.Quantity (GeV)
           The best-fit noise parameter.
           If not specified then the maximize_likelihood function is called.
    
    llmax : (optional) float
            The maximum value of the loglikelihood function.
            If not specified, this is computed, either using the specified gmax and smax
            or from the call to maximize_likelihood.

    g_scale : (optional) astropy.Quantity (GeV^-1 or natural equivalent)
              Rough scale/estimate for the coupling g.
              Used for initial conditions in the optimization routine.
              Defaults to 1.e-10*u.GeV**-1

    s_scale: (optional) astropy.Quantity (GeV)
             Rough scale/estimate for the noise parameter s.
             Used for initial conditions in the optimization routines.             
             Defaults to 1.e-34*u.GeV

    Returns
    --------------
    glim : astropy.Quantity (GeV^-1)
           The upper limit on the coupling constant for the model at the specified confidence
    """
    x = np.concatenate((A,B))

    if gmax is None or smax is None:
        gmax, smax, llmax = maximize_likelihood(A, B, c, g_scale=g_scale, s_scale=s_scale)
    elif llmax is None:
        llmax = loglikelihood(x, gmax, smax, c)

    ts_critical = stats.chi2.ppf(1.-2.*(1.-confidence), df=1)        
    ll_critical = llmax - 0.5*ts_critical

    def f_root(x):
        log10g = x
        s, ll = profile_likelihood(A, B, c, 10.**log10g*u.GeV**-1, s_scale)
        return ll-ll_critical

    log10gmax = np.log10(gmax.to_value(u.GeV**-1))

    res = opt.root_scalar(f_root, bracket=[log10gmax, -4.])

    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if fmin>0.1:
        print("WARNING: Root finding may have failed.")
    return 10.**log10glim*u.GeV**-1
        
