"""
This module for the "deterministic" time domain analysis.

This analysis is never fully correct, but only makes sense
when the coherence time is much longer than the experimental integration time,
so that incoherent fluctuations can be ignored.
This comparable to the analysis modules coherent.py and coherent_1d.py
"""

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import i0
from axionpy import units as u
from axionpy.maxwell import _vo
from axionpy.constants import _rhodm
from axionpy.analysis.coherent_1d import compute_coefficients # compute coefficients works the same as in the coherent_1d analysis

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
    return 

def loglikelihood(az, bz, g, s):
    """
    Computes the loglikelihood of the data given the model parameters.

    Parameters
    ----------------
    az, bz : astropy.Quantity (GeV)
             The Az, Bz fit coefficients for the z-component of the xyz basis.
           
    g : astropy.Quantity (GeV^-1 or natural equivalent)
        The axion coupling constant.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    Returns
    ------------------
    ll : float
         The log-likelihood evaluated with the given data and model parameters.
    """
    amp = np.sqrt(az**2 + bz**2)
    amp_pred = u.convert(g*np.sqrt(2.*_rhodm)*_vo, u.GeV)

    # az = az_true + noise
    # bz = bz_true + noise
    # model parameters determine sqrt(az_true**2 + bz_true**2) uniquely
    # The distribution of sqrt(az**2 + bz**2) is a Rice distribution
    # see, e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html

    #l = stats.rice.pdf((amp/s).to_value(u.dimensionless_unscaled), (amp_pred/s).to_value(u.dimensionless_unscaled))
    #return np.log(l)
    ll = np.log((amp/s**2).to_value(u.GeV**-1)) - 0.5*((amp**2 + amp_pred**2)/s**2).to_value(u.dimensionless_unscaled) + np.log(i0((amp*amp_pred/s**2).to_value(u.dimensionless_unscaled)))
    return ll

def maximize_likelihood(az, bz, s, g_scale=None):
    """
    Maximizes the likelihood with respect to the axion coupling g,
    asssuming a fixed noise parameter s.
    
    Parameters:
    ------------------------
    az, bz : astropy.Quantity (GeV)
             The Az, Bz fit coefficients for the z-component of the xyz basis.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    g_scale : (optional) astropy.Quantity (GeV^-1 or natural equivalent)
              Rough scale/estimate for the coupling g.
              Used for initial conditions in the optimization routine.
              Defaults to sqrt(az**2 + bz**2)/sqrt(2*rhodm*sigma**2)

    Returns:
    ------------------------
    g : astropy.Quantity (GeV^-1)
        The best-fit coupling constant.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """

    def f_to_minimize(log10g):
        ll = loglikelihood(az, bz, 10.**log10g*u.GeV**-1, s)
        if np.isfinite(ll):
            return -1.*ll
        #print("WARNING: non-finite ll: "+str(ll))
        return -1.*ll

    if g_scale is None:
        p0 = np.log10(u.convert(np.sqrt(az**2 + bz**2)/np.sqrt(2.*_rhodm*_vo**2), u.GeV**-1, value=True))
    else:
        p0 = np.log10(g_scale.to_value(u.GeV**-1))
    res = opt.minimize(f_to_minimize, p0, bounds=[[None,0.0]])
    log10g = res.x
    maxll = -1.*res.fun
    if not np.isfinite(maxll):
        raise Exception("ERROR: maxll is not finite")
    return 10.**log10g*u.GeV**-1, maxll
    
def frequentist_upper_limit(az, bz, s, confidence=0.95, gmax=None, llmax=None, g_scale=None):
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
    az, bz : astropy.Quantity (GeV)
             The Az, Bz fit coefficients for the z-component of the xyz basis.

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
              Defaults to sqrt(az**2 + bz**2)/sqrt(2*rhodm*sigma**2)

    Returns
    --------------
    glim : astropy.Quantity (GeV^-1)
           The upper limit on the coupling constant for the model at the specified confidence
    """
    
    if gmax is None:
        gmax, llmax = maximize_likelihood(az, bz, s)
    elif llmax is None:
        llmax = loglikelihood(az, bz, gmax, s)

    ts_critical = stats.chi2.ppf(1.-2.*(1.-confidence), df=1)
    ll_critical = llmax - 0.5*ts_critical

    def f_root(log10g):
        ll = loglikelihood(az, bz, 10.**log10g*u.GeV**-1, s)
        return ll-ll_critical

    log10gmax = np.log10(gmax.to_value(u.GeV**-1))
    res = opt.root_scalar(f_root, bracket=[log10gmax, -6.])

    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if not np.isfinite(fmin):
        raise Exception("ERROR: fmin is not finite")

    if fmin>0.1:
        print("WARNING: root finding may have failed")
    return 10.**log10glim*u.GeV**-1
