"""
This module for the full time domain analysis.
"""
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
    
    M = np.transpose([ np.cos(mat), np.sin(mat) ]) # (N, 2)
    coeffs, resid, rank, s = np.linalg.lstsq(M, yval, rcond=None)
    A, B = coeffs
    return A*unit, B*unit
        
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
    raise Exception("ERROR: NOT IMPLEMENTED")

def maximize_likelihood(m, t, x, xerr=None, g=None):
    """
    m: axion mass [astropy.Quantity in equivalent units to Angular Frequency]
    t: time series [(N,) astropy.Quantity in units of Time]
    x: data [ (2, N) astropy.Quantity in units of GeV ]
    xerr: uncertainty [ optional scalar or (2,N) array astropy.Quantity in units of GeV]
    g: coupling [opional fixed value of coupling, for constrained optimization]

    Returns
    """
    raise Exception("ERROR: NOT IMPLEMENTED")

def frequentist_upper_limit(m, t, x, xerr=None, confidence=0.95):
    """
    m: axion mass [astropy.Quantity in equivalent units to Angular Frequency]
    t: time series [(N,) astropy.Quantity in units of Time]
    x: data [ (2, N) astropy.Quantity in units of GeV ]
    xerr: uncertainty [ optional scalar or (2,N) array astropy.Quantity in units of GeV]
    g: coupling [opional fixed value of coupling, for constrained optimization]
    """
    raise Exception("ERROR: NOT IMPLEMENTED")

