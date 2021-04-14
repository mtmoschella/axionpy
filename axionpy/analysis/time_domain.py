from abc import ABC

class TimeDomainAnalysis:
    """
    This is a class for a full analysis.
    """
    
    def __init__(self):
        pass

    def loglikelihood(g, m, t, x, xerr):
        """
        Computes the loglikelihood of the data given model parameters.

        g: axion coupling [astropy.Quantity in equivalent units to GeV^-1]
        m: axion mass [astropy.Quantity in equivalent units to Angular Frequency]
        t: time series [(N,) astropy.Quantity in units of Time]
        x: data [ (2, N) astropy.Quantity in units of GeV ]
        xerr: data uncertainty [ scalar or (2,N) array astropy.Quantity in units of GeV]

        Returns a float
        """
        pass

    def maximize_likelihood(m, t, x, xerr=None, g=None):
        """
        m: axion mass [astropy.Quantity in equivalent units to Angular Frequency]
        t: time series [(N,) astropy.Quantity in units of Time]
        x: data [ (2, N) astropy.Quantity in units of GeV ]
        xerr: uncertainty [ optional scalar or (2,N) array astropy.Quantity in units of GeV]
        g: coupling [opional fixed value of coupling, for constrained optimization]

        Returns
        """
        pass

    def frequentist_upper_limit(m, t, x, xerr=None, confidence=0.95):
        """
        m: axion mass [astropy.Quantity in equivalent units to Angular Frequency]
        t: time series [(N,) astropy.Quantity in units of Time]
        x: data [ (2, N) astropy.Quantity in units of GeV ]
        xerr: uncertainty [ optional scalar or (2,N) array astropy.Quantity in units of GeV]
        g: coupling [opional fixed value of coupling, for constrained optimization]
        """
        
    def compute_coefficients(self):
        """
        Compute
        """
