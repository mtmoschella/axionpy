"""
This module for the coherent time domain analysis.

Everything here only makes sense in the limit of very long coherence times.
"""
import numpy as np
import scipy.optimize as opt
from scipy import stats
from axionpy import units as u
from axionpy.axis import Axis
from axionpy.constants import _rhodm
from axionpy.maxwell import _sigma, _vo

def compute_coefficients(m, t, y, **kwargs):
    """
    Compute the (Ax,Bx,Ay,By,Az,Bz) coefficients for the given time series of data.
    Physically, this only makes sense if the total time interval
    is longer than 1 day and much shorter than the axion coherence time.

    Parameters
    ---------------
    m : astropy.Quantity (frequency)
        The axion mass. 
        NOTE: This is treated as an angular frequency.

    t : (N,) astropy.Quantity (time)
        The time series.

    y : (N,) astropy.Quantity
        The observed data. Can have arbitrary astropy units.

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

    Returns
    ---------------
    array([Ax, Ay, Az]), array([Bx, By, Bz]) : astropy.Quantity
           The A and B coefficients such that 
           y = sum_i[ (Ai*cos(m*t) - Bi*sin(m*t))*mi is the least-squares solution
           where mi is the measurement axis projected along the ith xyz basis vector.
           Output is an astropy.Quantity with the same units as y.
    
    sigma : astropy.Quantity
            The Gaussian uncertainty on the coefficients.
            Output is an astropy.Quantity with the same units as y.
    """
    # check if valid time series
    T = np.amax(t)-np.amin(t)
    if T>1.e5/m:
        print("WARNING: compute_coefficients only makes physical sense on short timescales.")

    if 'components' in kwargs:
        components = kwargs['components']
    else:
        if 'axis' in kwargs:
            axis = kwargs['axis']
        elif 'lat' in kwargs and 'lon' in kwargs and 'theta' in kwargs and 'phi' in kwargs:
            axis = Axis(kwargs['lat'], kwargs['lon'], kwargs['theta'], kwargs['phi'])
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
        
    unit = y.unit
    yval = y.to_value(unit)
    mt = (m*t).to_value(u.dimensionless_unscaled) # radians
    
    M = np.transpose([ np.cos(mt)*components[0], -1.*np.sin(mt)*components[0], np.cos(mt)*components[1], -1.*np.sin(mt)*components[1], np.cos(mt)*components[2], -1.*np.sin(mt)*components[2] ]) # (N, 6)
    coeffs, resid, rank, s = np.linalg.lstsq(M, yval, rcond=None)
    Ax, Bx, Ay, By, Az, Bz = coeffs
    sigma = np.sqrt(2.*resid[0]/len(t))
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    return A*unit, B*unit, sigma*unit

def loglikelihood(a, b, g, s):
    """
    Computes the loglikelihood of the data given the model parameters.

    Parameters
    ----------------
    a, b : (3,) astropy.Quantity (GeV)
           The A, B fit coefficients for each component of the xyz basis.
           
    g : astropy.Quantity (GeV^-1 or natural equivalent)
        The axion coupling constant.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    Returns
    ------------------
    ll : float
         The log-likelihood evaluated with the given data and model parameters.
    """
    s_eff = np.array([ u.convert(np.sqrt(g**2*_rhodm*_sigma**2), u.GeV, value=True), u.convert(np.sqrt(g**2*_rhodm*_sigma**2), u.GeV, value=True), u.convert(np.sqrt(g**2*_rhodm*(_sigma**2 + _vo**2)), u.GeV, value=True)])*u.GeV # (3,)

    chi2 = np.sum(((a**2 + b**2)/(s_eff**2 + s**2)).to_value(u.dimensionless_unscaled))
    logdet = np.sum(np.log(2.*np.pi*((s_eff**2 + s**2).to_value(u.GeV**2))**2))

    return -0.5*(chi2 + logdet)
    
def maximize_likelihood(a, b, s, g_scale=None):
    """
    Maximizes the likelihood with respect to the axion coupling g,
    asssuming a fixed noise parameter s.

    Parameters:
    ------------------------
    a,b : (3,) astropy.Quantity (GeV)
          The A, B fit coefficients for each component of the xyz basis.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    g_scale : (optional) astropy.Quantity (GeV^-1 or natural equivalent)
              Rough scale/estimate for the coupling g.
              Used for initial conditions in the optimization routine.
              Defaults to sqrt(az**2 + bz**2)/sqrt(rhodm*sigma**2)

    Returns:
    ------------------------
    g : astropy.Quantity (GeV^-1)
        The best-fit coupling constant.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """
    
    def f_to_minimize(log10g):
        ll = loglikelihood(a, b, 10.**log10g*u.GeV**-1, s)
        if np.isfinite(ll):
            return -1.*ll
        raise Exception("ERROR: non-finite ll")

    if g_scale is None:
        p0 = np.log10(u.convert(np.sqrt(a[-1]**2 + b[-1]**2)/np.sqrt(_rhodm*_sigma**2), u.GeV**-1, value=True))
    else:
        p0 = np.log10(g_scale.to_value(u.GeV**-1))
    res = opt.minimize(f_to_minimize, p0, bounds=[[None,0.0]])
    log10g = res.x
    maxll = -1.*res.fun
    return 10.**log10g*u.GeV**-1, maxll

def frequentist_upper_limit(a, b, s, confidence=0.95, gmax=None, llmax=None, g_scale=None):
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
    a,b : (3,) astropy.Quantity (GeV)
          The A, B fit coefficients for each component of the xyz basis.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    confidence : (optional) float
                 The requested confidence for the upper limit. 
                 Must satify 0.<confidence<1.
                 Defaults to 0.95

    gmax : (optional) astropy.Quantity (GeV^-1)
           The best-fit coupling constant. 
           If not specified then the maximize_likelihood function is called.
    
    llmax : (optional) float
            The maximum value of the loglikelihood function.
            If not specified, this is computed, either using the specified gmax and smax
            or from the call to maximize_likelihood.

    g_scale : (optional) astropy.Quantity (GeV^-1 or natural equivalent)
              Rough scale/estimate for the coupling g.
              Used for initial conditions in the optimization routine.
              Defaults to 1.e-10*u.GeV**-1

    Returns
    --------------
    glim : astropy.Quantity (GeV^-1)
           The upper limit on the coupling constant for the model at the specified confidence
    """
    if gmax is None:
        gmax, llmax = maximize_likelihood(a, b, s, g_scale=g_scale)
    elif llmax is None:
        llmax = loglikelihood(a, b, gmax, s)

    ts_critical = stats.chi2.ppf(1.-2.*(1.-confidence), df=1)        
    ll_critical = llmax - 0.5*ts_critical

    def f_root(log10g):
        ll = loglikelihood(a, b, 10.**log10g*u.GeV**-1, s)
        return ll-ll_critical

    log10gmax = np.log10(gmax.to_value(u.GeV**-1))
    res = opt.root_scalar(f_root, bracket=[log10gmax, -6.])
    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if fmin>0.1:
        print("WARNING: Root finding may have failed.")
    return 10.**log10glim*u.GeV**-1

