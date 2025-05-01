import numpy as np
from scipy.special import erf

# Constants
GAS_CONST = 1.98720425864016  # in cal/(K*mol)
AVOGADRO = 6.02214076e23  # molecules/mol
STD_CONVERSION = 1.0e27 / AVOGADRO  # Ang^3/molecule at 1 mole/L


def initial():
    pass  # Placeholder for initial setup function


def final():
    pass  # Placeholder for finalization function


def get_next_arg(string_list):
    if string_list:
        return string_list.pop(0), True
    return None, False


def hfix(args):
    # Initialize variables ri fi ro fo temp
    ri,fi,ro,fo,temp = [float(arg) for arg in args]

    if fi == 0.0:
        fi = 1.0
        ri = 0.0
    if fo == 0.0:
        fo = 1.0

    # Output restraint parameters with aligned formatting
    print(f'{"Inner Flat-Bottom Radius:":35} {ri:.4f} Ang')
    print(f'{"Inner Force Constant:":35} {fi:.4f} Kcal/mole/Ang^2')
    print(f'{"Outer Flat-Bottom Radius:":35} {ro:.4f} Ang')
    print(f'{"Outer Force Constant:":35} {fo:.4f} Kcal/mole/Ang^2')
    print(f'{"System Temperature Value:":35} {temp:.4f} Kelvin\n')

    kt = temp * GAS_CONST  # RT in cal/mol
    kB = 0.0019872041  # Boltzmann constant in kcal/mol/K
    kt = kB * temp      # Thermal energy (kcal/mol)
    vol, dvol = compute_volume_integrals(ri, fi, ro, fo, temp)

    # Output analytical volume integral and dVol/dT with aligned formatting
    print(f'{"Analytical Volume Integral:":35} {vol:.4f} Ang^3')
    print(f'{"Analytical dVol/dT Value:":35} {dvol:.4f} Ang^3/K\n')

    # Thermodynamic values
    dg = -kt * np.log(vol / STD_CONVERSION)
    ds = -dg / temp + kt * dvol / vol
    dh = dg + temp * ds

    print(f'{"Restraint Free Energy:":35} {dg:.4f} Kcal/mole')
    print(f'{"Restraint Entropy Value:":35} {ds:.4f} Kcal/mole/K')
    print(f'{"Restraint Enthalpy Value:":35} {dh:.4f} Kcal/mole')
    print(f'{"Restraint -T deltaS Value:":35} {-temp * ds:.4f} Kcal/mole')


def compute_volume_integrals(ri, fi, ro, fo, temp):
    """
    Computes the analytical restraint volume integral and its temperature derivative.

    Returns:
    -------
    vol : float
        Volume integral (Å³)
    dvol : float
        Derivative of volume integral with respect to temperature (Å³/K)
    """
    # Constants
    kB = 0.0019872041  # Boltzmann constant in kcal/mol/K
    kt = kB * temp      # Thermal energy (kcal/mol)
    # Instead of passing kt as an external parameter, it's now computed inside the function using the Boltzmann constant (kB) and the provided temperature (temp)

    # Analytical evaluation of the restraint volume integral
    # Handle ri = 0 to avoid unnecessary computations
    if ri == 0.0:
        v1 = 0.0 #avoid unnecessary calculations
    else:
        term1_v1 = 2.0 * np.pi * ri * (-2.0 + np.exp(-ri ** 2 * fi / kt)) * kt / fi
        term2_v1 = np.sqrt(kt * (np.pi / fi) ** 3) * (2.0 * fi * ri ** 2 + kt) * erf(ri * np.sqrt(fi / kt))
        v1 = term1_v1 + term2_v1

    # Volume integral from inner and outer radii
    v2 = (4.0 / 3.0) * np.pi * (ro ** 3 - ri ** 3)

    # Handle ro = 0 to simplify v3
    if ro == 0.0:
        term1_v3 = np.sqrt(kt * (np.pi / fo) ** 3) * kt
    else:
        term1_v3 = np.sqrt(kt * (np.pi / fo) ** 3) * (2.0 * fo * ro ** 2 + kt + 4.0 * ro * np.sqrt(kt * fo / np.pi))
    v3 = term1_v3

    # Total volume
    vol = v1 + v2 + v3

    # Debug: Print intermediate volume components
    # print(f"v1: {v1}, v2: {v2}, v3: {v3}")

    # Derivatives for dVol/dT calculation
    # Since v2 does not depend on temperature, its derivative is zero

    # Derivative of v1 with respect to temperature
    if ri == 0.0:
        dv1 = 0.0 #since there are no temperature-dependent terms
    else:
        term1_dv1 = 2.0 * np.pi * ri ** 3 * np.exp(-ri ** 2 * fi / kt) / temp
        term2_dv1 = 2.0 * np.pi * ri * (-2.0 + np.exp(-ri ** 2 * fi / kt)) * kt / (fi * temp)
        term3_dv1 = 0.5 * np.sqrt((np.pi / fi) ** 3) * np.sqrt(kt) * (2.0 * fi * ri ** 2 + kt) * erf(ri * np.sqrt(fi / kt)) / temp
        term4_dv1 = - np.pi * ri * np.exp(-ri ** 2 * fi / kt) * (2.0 * fi * ri ** 2 + kt) / (fi * temp)
        term5_dv1 = np.sqrt((kt * np.pi / fi) ** 3) * erf(ri * np.sqrt(fi / kt)) / temp
        dv1 = term1_dv1 + term2_dv1 + term3_dv1 + term4_dv1 + term5_dv1

    # Derivative of v3 with respect to temperature
    if ro == 0.0:
        # Only the term involving kt remains
        dv3 = 1.5 * (kt * np.pi / fo) ** 1.5 / temp
    else:
        term1_dv3 = np.sqrt(kt * (np.pi / fo) ** 3) * fo * ro ** 2 / temp
        term2_dv3 = 4.0 * kt * (np.pi / fo) * ro / temp
        term3_dv3 = 1.5 * np.sqrt((kt * np.pi / fo) ** 3) / temp
        dv3 = term1_dv3 + term2_dv3 + term3_dv3

    # Total derivative
    dvol = dv1 + dv3

    return vol, dvol


def freefix(args):
    initial()
    method = 'HARMONIC'

    # Call corresponding method based on the method type
    if method == 'HARMONIC':
        print("Calculating energy of HARMONIC restraints. BORESCH restraints not implemented yet.")
        hfix(args)
    elif method == 'BORESCH':
        print("BORESCH method not implemented yet.")

    final()


# Example of how to run the code
if __name__ == "__main__":
    import sys

    # Simulating command line arguments, replace with actual input as needed
    # Example usage: python script.py ri fi ro fo temp
    # For instance: ['3.0', '1.0', '3.5', '1.0', '298.0']
    input_args = sys.argv[1:]
    freefix(input_args)
