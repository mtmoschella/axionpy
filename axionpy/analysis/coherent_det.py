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
    amp = np.sqrt(az**2 + bz**2).to_value(u.GeV)
    amp_pred = u.convert(g*np.sqrt(_rhodm)*vo, u.GeV, value=True)

    chi2 = (amp-amp_pred)**2/s.to_value(u.GeV)**2
    logdet = np.log(2.*np.pi*s.to_value(u.GeV)**2)
    return -0.5*(chi2 + logdet)

def maximize_likelihoood(az, bz, s):
    """
    Maximizes the likelihood with respect to the axion coupling g,
    asssuming a fixed noise parameter s.
    
    Parameters:
    ------------------------
    az, bz : astropy.Quantity (GeV)
             The Az, Bz fit coefficients for the z-component of the xyz basis.

    s : astropy.Quantity (GeV)
        The constant uncertainty on the A and B coefficients.

    Returns:
    ------------------------
    g : astropy.Quantity (GeV^-1)
        The best-fit coupling constant.

    maxll : float
            The value of maximum value of the loglikelihood function.
    """
    
    g = _g(az,bz)

    # compute the maximum value of the likelihood
    maxll = loglikelihood(az, bz, g, s)

    return g, maxll
    
def frequentist_upper_limit(az, bz, s, confidence=0.95, gmax=None, llmax=None):
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

    # NOTE: The likelihood is a 1D Gaussian function, this could be implemented analytically
    def f_root(log10g):
        ll = loglikelihood(az, bz, 10.**log10g*u.GeV**-1, s)
        return ll-ll_critical

    log10gmax = np.log10(gmax.to_value(u.GeV**-1))
    res = opt.root_scalar(f_root, bracket=[log10gmax, -6.])

    log10glim = res.root
    fmin = np.absolute(f_root(log10glim))
    if fmin>0.1:
        print("WARNING: root finding may have failed")
    return 10.**log10glim*u.GeV**-1
