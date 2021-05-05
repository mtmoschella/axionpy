"""
This module for the full time domain analysis.
"""
import numpy as np
import astropy.units as u
import natural_units as nat
import scipy.optimize as opt
from scipy import stats
from ..maxwell import cov

_rhodm = 0.3*u.GeV/u.cm**3

def compute_coefficients(m, t, y):
    """
    Compute the A and B coefficients for the given time series of data.
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
        The observed data.

    Returns
    ---------------
    A, B : astropy.Quantity
           The A and B coefficients such that 
           A*cos(m*t) + B*sin(m*t) is the least-squares solution for the data y
           Output is an astropy.Quantity with the same units as y
    """
    # check if valid time series
    T = np.amax(t)-np.amin(t)
    if T > 1.e5/m or T>0.1*u.day:
        print("WARNING: compute_coefficients only makes physical sense on short timescales.")

    unit = y.unit
    yval = y.to_value(unit)
    mt = (m*t).to_value(u.dimensionless_unscaled) # radians
    
    M = np.transpose([ np.cos(mt), np.sin(mt) ]) # (N, 2)
    coeffs, resid, rank, s = np.linalg.lstsq(M, yval, rcond=None)
    A, B = coeffs
    return A*unit, B*unit
        
def loglikelihood(x, g, s, c):
    """
    Computes the loglikelihood of the data given model parameters.
    
    g: axion coupling [astropy.Quantity in equivalent units to GeV^-1]
    s: data uncertainty [astropy.Quantity in GeV]
    x: data [ (2N,) astropy.Quantity in GeV]
    c: CovMatrix object containing the covariance matrix

    Returns a float
    """
    geff = nat.convert(np.sqrt(_rhodm)*g, u.GeV) # rescale for covariance matrix
    chi2 = c.chi2(geff.to_value(u.GeV), s.to_value(u.GeV), x.to_value(u.GeV))
    logdet = c.logdet(geff.to_value(u.GeV), s.to_value(u.GeV))
    return -0.5*(chi2 + logdet)

def maximize_likelihood(A, B, c, g_scale=1.0e-10*u.GeV**-1, s_scale=1.e-34*u.GeV):
    """
    A: (N,) astropy.Quantity in GeV
    B: (N,) astropy.Quantity in GeV
    c: CovMatrix object representing NxN matrix
    g_scale: rough scale for g in astropy GeV**-1
    s_scale: rough scale for s in astropy GeV
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
    A: (N,) astropy.Quantity in GeV
    B: (N,) astropy.Quantity in GeV
    c: CovMatrix object representing NxN matrix
    g: fixed value of coupling, astropy GeV^-1
    p0: initial guess for log10s
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

    try:
        res = opt.root_scalar(f_root, bracket=[log10gmax, -6.])
    except:
        print("WARNING: received error in Root Finding")
        raise 
        return 0.0*u.GeV**-1

    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if fmin>0.1:
        print("WARNING: Root finding may have failed.")
    return 10.**log10glim*u.GeV**-1
        
