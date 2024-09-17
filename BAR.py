from pymbar import other_estimators
import argparse
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from mdtraj import load_dcd
from intspan import intspan

# Argument parser for user-defined flags
parser = argparse.ArgumentParser(description='BAR analysis for OpenMM Alchemical Simulations')
parser.add_argument('--traj_i', required=True, type=str, help='DCD file for lambda i')
parser.add_argument('--traj_ip1', required=True, type=str, help='DCD file for lambda i+1')
parser.add_argument('--pdb_file', required=True, type=str, help='PDB file for the simulation')
parser.add_argument('--forcefield_file', required=True, type=str, help='Force field XML file')
parser.add_argument('--vdw_lambda_i', required=True, type=float, help='Lambda for van der Waals at state i')
parser.add_argument('--elec_lambda_i', required=True, type=float, help='Lambda for electrostatics at state i')
parser.add_argument('--vdw_lambda_ip1', required=True, type=float, help='Lambda for van der Waals at state i+1')
parser.add_argument('--elec_lambda_ip1', required=True, type=float, help='Lambda for electrostatics at state i+1')
parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
args = parser.parse_args()

# Helper function to set lambda values and update forces in the context
def set_lambda_values(context, vdw_lambda, elec_lambda, vdwForce, multipoleForce, alchemical_atoms):
    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
    
    for i in alchemical_atoms:
        params = multipoleForce.getMultipoleParameters(i)
        charge, dipole, quadrupole, polarizability = params[0], params[1], params[2], params[-1]
        charge = charge * elec_lambda
        dipole = [d * elec_lambda for d in dipole]
        quadrupole = [q * elec_lambda for q in quadrupole]
        polarizability = polarizability * elec_lambda
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)

    vdwForce.updateParametersInContext(context)
    multipoleForce.updateParametersInContext(context)

def setup_simulation(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms):
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(forcefield_file)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*femtosecond)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)

    numForces = system.getNumForces()
    vdwForce, multipoleForce = None, None
    for i in range(numForces):
        force = system.getForce(i)
        if isinstance(force, AmoebaVdwForce):
            vdwForce = force
        elif isinstance(force, AmoebaMultipoleForce):
            multipoleForce = force

    if vdwForce is None or multipoleForce is None:
        raise ValueError("AmoebaVdwForce or AmoebaMultipoleForce not found in the system.")

    return context, vdwForce, multipoleForce, system, pdb

def compute_work(traj_file, context, pdb, vdw_lambda_1, vdw_lambda_2, elec_lambda_1, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms):
    traj = load_dcd(traj_file, pdb_file)
    energy11, energy12, energy22, energy21 = [], [], [], []

    # Forward work: lambda i -> lambda i+1
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy11 = []
    for frame in range(len(traj)):
        context.setPositions(traj.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy11.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy12 = []
    for frame in range(len(traj)):
        context.setPositions(traj.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy12.append(potential_energy)
    
    forward_work = np.array(energy12) - np.array(energy11)

    # Reverse work: lambda i+1 -> lambda i
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy22 = []
    for frame in range(len(traj)):
        context.setPositions(traj.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy22.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy21 = []
    for frame in range(len(traj)):
        context.setPositions(traj.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy21.append(potential_energy)

    reverse_work = np.array(energy21) - np.array(energy22)

    return forward_work, reverse_work

# Main execution
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
alchemical_atoms = list(intspan(args.alchemical_atoms))
nonbonded_method = {'NoCutoff': NoCutoff, 'CutoffNonPeriodic': CutoffNonPeriodic, 'PME': PME, 'Ewald': Ewald}[args.nonbonded_method]
nonbonded_cutoff = 1.0 * nanometer

context, vdwForce, multipoleForce, system, pdb = setup_simulation(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms)

# Forward and reverse work calculation
forward_work, reverse_work = compute_work(args.traj_i, context, pdb, args.vdw_lambda_i, args.vdw_lambda_ip1, args.elec_lambda_i, args.elec_lambda_ip1, vdwForce, multipoleForce, alchemical_atoms)

# Perform BAR analysis
bar_results = other_estimators.bar(forward_work, reverse_work)

# The `Delta_f` is the free energy difference between two states (lambda_i and lambda_i+1).
# In this case, it's the change in free energy due to modifying the van der Waals and/or electrostatic parameters 
# between two adjacent lambda values. The free energy difference (ΔF) helps quantify how much the system's energy changes 
# when transitioning between these states in the alchemical transformation.
#
# The `dDelta_f` represents the uncertainty (or standard error) in the estimated free energy difference.
# It provides an estimate of how much error or variation is present in the `Delta_f` calculation due to sampling noise.
# Smaller values of `dDelta_f` indicate a more accurate and reliable estimate of the free energy difference.


print(f"Free energy difference: {bar_results['Delta_f']} ± {bar_results['dDelta_f']} kJ/mol")
