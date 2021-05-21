"""
This module for the "deterministic" time domain analysis.

This analysis is never fully correct, but only makes sense
when the coherence time is much longer than the experimental integration time,
so that incoherent fluctuations can be ignored.
"""

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from axionpy import units as u
from axionpy.maxwell import _vo
from axionpy.constants import _rhodm

def _g(a, b):
    """
    Given the amplitudes a and b, return the coupling g.

    Parameters
    --------------------
    a, b : astropy.Quantity (GeV)
           The axion amplitudes.

    Returns
    --------------------
    g : astropy.Quantity (GeV^-1)
        The axion coupling constant
    """
    return u.convert(np.sqrt(a**2 + b**2)/(np.sqrt(_rhodm)*_vo), u.GeV**-1)

def _p(a, b):
    """
    Given the amplitudes a and b, return the phase p.

    Parameters
    --------------------
    a, b : astropy.Quantity (GeV)
           The axion amplitudes.

    Returns
    --------------------
    p : float
        The axion phase in radians.
    """
    return np.arctan2(b.to_value(u.GeV), a.to_value(u.GeV))

def _a(g, p):
    """
    Given the axion coupling g and phase p, return the cosine amplitude a

    Parameters
    --------------------
    g : astropy.Quantity (GeV^-1)
        The axion coupling constant
    
    p : float
        The axion phase in radians.

    Returns
    --------------------
    a : astropy.Quantity (GeV)
        The axion cosine amplitude.
    """
    return u.convert(g*np.cos(p)*np.sqrt(_rhodm)*_vo, u.GeV)

def _b(g, p):
    """
    Given the axion coupling g and phase p, return the sine amplitude b

    Parameters
    --------------------
    g : astropy.Quantity (GeV^-1)
        The axion coupling constant
    
    p : float
        The axion phase in radians.

    Returns
    --------------------
    b : astropy.Quantity (GeV)
        The axion sine amplitude.
    """
    return u.convert(g*np.sin(p)*np.sqrt(_rhodm)*_vo, u.GeV)

def loglikelihood(x, mt, mz, g, p, s):
    """
    Computes the loglikelihood of the data given the model parameters.

    Parameters
    ------------------
    x : (N,) astropy.Quantity (GeV)
        The time series of raw data
    
    mt : (N,) array-like
         The dimensionless time series m*t 
         where t is the time 
         and m is the axion mass

    mz : (N,) array-like
         The component of the sensitive axis along the z direction.
         The deterministic analysis assumes that the axion field only
         has gradient in the z direction (parallel to vsun)

    g : astropy.Quantity (GeV^-1 or natxural equivalent)
        The axion coupling constant.

    p : float
        The axion phase

    s : astropy.Quantity (GeV)
        The noise parameter, a constant white noise uncertainty on each data point

    Returns
    ------------------
    ll : float
         The log-likelihood evaluated with the given data and model parameters.
    """
    a = _a(g,p)
    b = _b(g,p)
    x_pred = (a*np.cos(mt) + b*np.sin(mt))*mz # exact model prediction
    chi2 = np.sum((((x-x_pred)/s).to_value(u.dimensionless_unscaled))**2)
    logdet = 2.*len(x)*np.log(s.to_value(u.GeV))
    return -0.5*(chi2 + logdet)

def maximize_likelihoood_deterministic(x, mt, mz):
    """
    Maximizes the likelihood with respect to 
    the axion coupling g, phase p, and the noise parameter s

    Parameters
    ------------------
    x : (N,) astropy.Quantity (GeV)
        The time series of raw data
    
    mt : (N,) array-like
         The dimensionless time series m*t 
         where t is the time 
         and m is the axion mass

    mz : (N,) array-like
         The component of the sensitive axis along the z direction.
         The deterministic analysis assumes that the axion field only
         has gradient in the z direction (parallel to vsun)
    
    Returns:
    ------------------------
    g : astropy.Quantity (GeV^-1)
        The best-fit coupling constant.

    p : float
        The best-fit phase in radians.

    s : astropy.Quantity (GeV)
        The best-fit noise parameter.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """
    M = np.transpose([ np.cos(mt)*mz, np.sin(mt)*mz])

    # get (a, b, s) in GeV via least-squares solution
    (a,b), resid, rank, sing = np.linalg.lstsq(M, x.to_value(u.GeV), rcond=None)
    s = np.sqrt(resid[0]/len(x))*u.GeV
    a *= u.GeV
    b *= u.GeV
    
    # convert (a, b) to (g, p)
    g = _g(a,b)
    p = _p(a,b)

    # compute the maximum value of the likelihood
    maxll = loglikelihood(x, mt, mz, g, p, s)

    return g, p, s, maxll

def profile_likelihood(x, mt, mz, g, p0=1.):
    """
    Maximizes the likelihood with respect to 
    the axion phase p, and the noise parameter s
    for a given value of the axion coupling g

    Parameters
    ------------------
    x : (N,) astropy.Quantity (GeV)
        The time series of raw data
    
    mt : (N,) array-like
         The dimensionless time series m*t 
         where t is the time 
         and m is the axion mass

    mz : (N,) array-like
         The component of the sensitive axis along the z direction.
         The deterministic analysis assumes that the axion field only
         has gradient in the z direction (parallel to vsun)

    g : astropy.Quantity (GeV^-1)
        The fixed value of the axion coupling.

    p0 : (optional) float
         The initial guess for the axion phase in radians.

    Returns:
    ------------------------
    p : float
        The best-fit phase in radians.

    s : astropy.Quantity (GeV)
        The best-fit noise parameter.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """

    amp = u.convert(g*np.sqrt(_rhodm)*_vo, u.GeV, value=True) # precompute unit conversion for efficiency
    def f_to_minimize(p):
        a = amp*np.cos(p)
        b = amp*np.sin(p)
        x_pred = (a*np.cos(mt) + b*np.sin(mt))*mz
        return np.sum((x.to_value(u.GeV)-x_pred)**2) # ssr in GeV

    if amp==0.0:
        p = 0.0
        residual = np.sum(x.to_value(u.GeV)**2)
    else:
        res = opt.minimize(f_to_minimize, p0)
        p = res.x
        residual = res.fun

    s = np.sqrt(residual/len(X))*u.GeV
    maxll = loglikelihood(x, mt, mz, amp*np.cos(p)*u.GeV, amp*np.sin(p)*u.GeV, s)
    
    return p, s, maxll
    
def frequentist_upper_limit(x, mt, mz, confidence=0.95, gmax=None, pmax=None, smax=None, llmax=None, p0=1.0):
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

    Parameters
    ------------------
    x : (N,) astropy.Quantity (GeV)
        The time series of raw data
    
    mt : (N,) array-like
         The dimensionless time series m*t 
         where t is the time 
         and m is the axion mass

    mz : (N,) array-like
         The component of the sensitive axis along the z direction.
         The deterministic analysis assumes that the axion field only
         has gradient in the z direction (parallel to vsun)

    confidence : (optional) float
                 The requested confidence for the upper limit. 
                 Must satify 0.<confidence<1.
                 Defaults to 0.95

    gmax : (optional) astropy.Quantity (GeV^-1)
           The best-fit coupling constant. 
           If not specified then the maximize_likelihood function is called.

    pmax : (optional) float
           The best-fit phase parameter.
           If not specified then the maximize_likelihood function is called.

    smax : (optional) astropy.Quantity (GeV)
           The best-fit noise parameter.
           If not specified then the maximize_likelihood function is called.
    
    llmax : (optional) float
            The maximum value of the loglikelihood function.
            If not specified, this is computed, either using the specified gmax and smax
            or from the call to maximize_likelihood.

    p0 : (optional) float
         The initial guess for the axion phase in radians.

    Returns:
    ------------------------
    glim : astropy.Quantity (GeV^-1)
           The upper limit on the coupling constant for the model at the specified confidence
    """
    
    if gmax is None or pmax is None or smax is None:
        gmax, pmax, smax, maxll = maximize_likelihood(x, mt, mz)
    elif llmax is None:
        llmax = loglikelihood(x, mt, mz, gmax, pmax, smax)

    ts_critical = stats.chi2.ppf(1.-2.*(1.-confidence), df=1)
    ll_critical = llmax - 0.5*ts_critical

    def f_root(log10g):
        p, s, ll = profile_likelihood(x, mt, mz, 10.**log10g*u.GeV**-1, p0=p0)
        return ll-ll_critical

    log10gmax = np.log10(gmax.to_value(u.GeV**-1))
    res = opt.root_scalar(f_root, bracket=[log10gmax, -6.])

    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if fmin>0.1:
        print("WARNING: root finding may have failed")
    return 10.**log10glim*u.GeV**-1
