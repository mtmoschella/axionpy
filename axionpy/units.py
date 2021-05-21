"""
This module is used for astropy-like units
with support for conversion to/from natural units 
where $\hbar=c=\epsilon_0=1$, with the convention that the unique
remaining 'natural' dimension is energy (eV by default).

In addition to natural unit conversion functions,
this module contains the entire astropy.units namespace.
For example, axionpy.units.kg is equivalent astropy.units.kg

This module is heavily based  off of github.com/mtmoschella/natural_units
"""
from astropy.units import * # import entire astropy.units namespace
from astropy.constants import c, hbar, eps0

def toNaturalUnits(x, output_unit=eV, verbose=False):
    """
    Converts the given physical astropy.Quantity to the natural units specified by output_unit.

    Parameters
    ------------------------------
    x : astropy.Quantity astropy.UnitBase
        The given quantity (can be array-like) but
    
    output_unit : (optional) astropy.UnitBase
                 The output base unit, must have physical_type=='energy'
                 Defaults to eV (electronVolt).

    verbose : (optional) boolean
              Run in verbose mode

    Returns 
    -------------------------------
    output : astropy.Quantity
             The input converted into natural units. 
             This will have unit of output_unit**d for some real number d.
    """

    # check input
    if not isinstance(x, Quantity):
        if isinstance(x, UnitBase):
            if verbose:
                print("WARNING: converting astropy unit to quantity by multiplying by 1.0")
            x *= 1.0
        else:
            raise Exception("ERROR: x must be either an astropy quantity or unit")
    assert isinstance(output_unit, UnitBase), "ERROR: output_unit must be a astropy unit"
    assert output_unit.physical_type=='energy', "ERROR: output_unit must be a unit of energy"
    
    unit = x.unit.decompose() # decompose into SI units (astropy default)
    bases = unit.bases 
    powers = unit.powers
    if set(bases)<=set([kg, m, s, A]): # (SI = MKS + Ampere)
        # get dimensions of kg, m, s, A units
        if kg in bases:
            i = powers[bases.index(kg)]
        else:
            i = 0
        if m in bases:
            j = powers[bases.index(m)]
        else:
            j = 0
        if s in bases:
            k = powers[bases.index(s)]
        else:
            k = 0
        if A in bases:
            l = powers[bases.index(A)]
        else:
            l = 0

        # convert to dimensions of hbar, c, eV, eps0 units
        # this comes from analytically solving the linear system of equations that you get from preserving dimensionality
        hbar_dim = j + k - 0.5*l
        c_dim = j - 2*i + 0.5*l
        E_dim = i - j - k + l
        eps0_dim = 0.5*l
        
        return (x/((hbar**hbar_dim)*(c**c_dim)*(eps0**eps0_dim))).to(output_unit**E_dim)

    else:
        raise Exception("ERROR: can only convert to natural units if MKS+A quantity")

def fromNaturalUnits(x, output_unit, value=False, verbose=False):
    """
    Converts the given (natural or physical) astropy Quantity in the physical units specified by output_unit.

    x: an astropy Quantity or Unit

    output_unit: an astropy UnitBase
                 must be naturally compatible with <x> or will raise AssertionError

    value: a boolean, if true, returns the output of astropy.to_value, otherwise return a astropy Quantity
    """

    if not isinstance(x, Quantity):
        if isinstance(x, UnitBase):
            if verbose:
                print("WARNING: converting astropy unit to quantity by multiplying by 1.0")
            x *= 1.0
        else:
            print("WARNING: converting ordinary scalar to an astropy dimensionless quantity")
            x *= dimensionless_unscaled
    assert isinstance(output_unit, UnitBase), "ERROR: output_unit must be a astropy unit"

    x = toNaturalUnits(x) # outputs in eV**n by default
    
    natunit = x.unit.decompose()
    natbases = natunit.bases
    natpowers = natunit.powers
    check_units = True
    
    unit = output_unit.decompose() # docompose into SI units (astropy default)
    bases = unit.bases
    powers = unit.powers
    
    if set(bases)<=set([kg, m, s, A]):
        # get dimensions of kg, m, s, A units
        if kg in bases:
            i = powers[bases.index(kg)]
        else:
            i = 0
        if m in bases:
            j = powers[bases.index(m)]
        else:
            j = 0
        if s in bases:
            k = powers[bases.index(s)]
        else:
            k = 0
        if A in bases:
            l = powers[bases.index(A)]
        else:
            l = 0

        # convert to dimensions of hbar, c, eV units
        # this comes from analytically solving the linear system of equations that you get from preserving dimensionality
        hbar_dim = j + k - 0.5*l
        c_dim = j - 2*i + 0.5*l
        E_dim = i - j - k + l
        eps0_dim = 0.5*l

        if check_units:
            # make sure that dim(x)==(energy)**E_dim
            # if not, then the specified output unit is not compatible
            if set(natbases)==set([]):
                # natunit is dimensionless
                assert E_dim==0.0, "ERROR: specified output_unit is not compatible with energy dimension 0"
            else:
                assert set(natbases)==set([kg, m, s]), "ERROR: natural units must be (energy)**n"
                kg_power = natpowers[natbases.index(kg)]
                m_power = natpowers[natbases.index(m)]
                s_power = natpowers[natbases.index(s)]
                assert kg_power==E_dim and m_power==2.0*E_dim and s_power==-2.0*E_dim, "ERROR: specified output_unit is not compatible with with energy dimension "+str(kg_power)

        output = (x*hbar**hbar_dim*c**c_dim*eps0**eps0_dim).to(output_unit)
        if value:
            return output.to_value(output_unit)
        else:
            return output
    else:
        raise Exception("ERROR: can only convert to natural units if MKS+A quantity")
        
def convert(x, unit, value=False, verbose=False):
    """
    Converts the given physical astropy.Quantity to the natural units specified by output_unit.

    Parameters
    ------------------------------
    x : astropy.Quantity astropy.UnitBase
        The given quantity (can be array-like) but
    
    output_unit : (optional) astropy.UnitBase
                 The output base unit, must have physical_type=='energy'
                 Defaults to eV (electronVolt).

    verbose : (optional) boolean
              Run in verbose mode

    Returns 
    -------------------------------
    output : astropy.Quantity
             The input converted into natural units. 
             This will have unit of output_unit**d for some real number d.
    """
    return fromNaturalUnits(x, unit, value, verbose)
