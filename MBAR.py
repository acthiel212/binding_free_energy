'''
python MBAR.py --traj_i output1.dcd --traj_ip1 output2.dcd --pdb_file g1.pdb --forcefield_file g1.xml --vdw_lambda_i 0 --elec_lambda_i 0 --vdw_lambda_ip1 0.4 --elec_lambda_ip1 0 --alchemical_atoms "0,2" --nonbonded_method NoCutoff
MBAR generalizes BAR by handling multiple states at once
it’s particularly useful when you have many intermediate states between two endpoints of an alchemical transformation
it estimates the free energy differences between all pairs of states

in this case, we have two states (lambda_i and lambda_ip1), and we're using MBAR to compute the free energy difference between them

workflow:
Trajectory loading:
load the simulation trajectories for state i (lambda = 0) and state i+1 (lambda = 0.4) from the DCD files

system setup: using openmm, we set up the system with the chosen force field, apply the lambda scaling for van der Waals (vdw_lambda) and electrostatic (elec_lambda) interactions for the alchemical atoms

computing potential energies: for each frame in the trajectory, we compute the potential energy of the system at both the starting (lambda_i) and ending (lambda_ip1) states
this gives us forward and reverse energy profiles.

work calculation: the difference in potential energies between the forward (energy12 - energy11) and reverse (energy21 - energy22) states is used to calculate the work required to transition between the states
these energy differences are then converted into reduced potentials (scaled by temperature)

MBAR analysis: MBAR computes the free energy difference Delta_f_ij and the uncertainty dDelta_f_ij between the two lambda states
'''

from pymbar import testsystems
import argparse
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from mdtraj import load_dcd
from pymbar import MBAR
from sys import stdout
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
    
    # Apply alchemical scaling to the multipole force
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
    pdb = PDBFile(pdb_file)  # This is the actual PDBFile object
    forcefield = ForceField(forcefield_file)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*femtosecond)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)

    # Get all forces in the system
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

    return context, vdwForce, multipoleForce, system, pdb  # Return the pdb object too



def compute_work(traj_file, context, pdb, vdw_lambda_1, vdw_lambda_2, elec_lambda_1, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms, temperature=300*kelvin):
    # Load the trajectory
    traj = load_dcd(traj_file, pdb_file)
    energy11, energy12, energy22, energy21 = [], [], [], []

    # Forward work: lambda i -> lambda i+1
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy11 = []
    for frame in range(len(traj)):
        # Set particle positions from the trajectory
        context.setPositions(traj.openmm_positions(frame))
        # Compute potential energy
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy11.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy12 = []
    for frame in range(len(traj)):
        # Set particle positions from the trajectory
        context.setPositions(traj.openmm_positions(frame))
        # Compute potential energy
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy12.append(potential_energy)
    
    forward_work = np.array(energy12) - np.array(energy11)

    # Reverse work: lambda i+1 -> lambda i
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy22 = []
    for frame in range(len(traj)):
        # Set particle positions from the trajectory
        context.setPositions(traj.openmm_positions(frame))
        # Compute potential energy
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy22.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy21 = []
    for frame in range(len(traj)):
        # Set particle positions from the trajectory
        context.setPositions(traj.openmm_positions(frame))
        # Compute potential energy
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy21.append(potential_energy)

    reverse_work = np.array(energy21) - np.array(energy22)
    print(f"Number of frames in traj_i: {len(energy11)}")
    print(f"Number of frames in traj_ip1: {len(energy21)}")

    forward_reduced = compute_reduced_potential(np.array(energy12) - np.array(energy11), temperature)
    reverse_reduced = compute_reduced_potential(np.array(energy21) - np.array(energy22), temperature)

    return forward_reduced, reverse_reduced
def compute_reduced_potential(energy, temperature=300*kelvin):
    kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA  # k_B in kJ/mol*K
    beta = 1 / (kB * temperature)
    return beta * energy

def prepare_mbar_input(forward_reduced, reverse_reduced):
    # Combine forward and reverse reduced energies
    u_kn = np.vstack([forward_reduced, reverse_reduced])

    # Total number of samples per state
    N_f = len(forward_reduced)  # Number of samples from the first state (lambda_i)
    N_r = len(reverse_reduced)  # Number of samples from the second state (lambda_ip1)

    # Ensure that the sum of N_k matches the total number of samples
    N_k = np.array([N_f, N_r])
    print(f"N_f (samples from state i): {N_f}")
    print(f"N_r (samples from state i+1): {N_r}")
    print(f"u_kn shape: {u_kn.shape}")

    if np.sum(N_k) != u_kn.shape[1]:
        raise ValueError("The sum of N_k must equal the total number of samples in u_kn.")

    return u_kn, N_k



# Main execution
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
alchemical_atoms = list(intspan(args.alchemical_atoms))
nonbonded_method = {'NoCutoff': NoCutoff, 'CutoffNonPeriodic': CutoffNonPeriodic, 'PME': PME, 'Ewald': Ewald}[args.nonbonded_method]
nonbonded_cutoff = 1.0 * nanometer  # Example cutoff

# Setup for forward (traj_i) and reverse (traj_ip1) work calculations
context, vdwForce, multipoleForce, system, pdb = setup_simulation(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms)

# Forward and reverse work calculation
forward_reduced, reverse_reduced = compute_work(args.traj_i, context, pdb, args.vdw_lambda_i, args.vdw_lambda_ip1, args.elec_lambda_i, args.elec_lambda_ip1, vdwForce, multipoleForce, alchemical_atoms)

# Prepare data for MBAR
u_kn, N_k = prepare_mbar_input(forward_reduced, reverse_reduced)

print(f"u_kn shape: {u_kn.shape}")  # Should be (2, 30)
print(f"Sum of N_k: {sum(N_k)}")  # Should be 30

print(f"Forward reduced shape: {forward_reduced.shape}")
print(f"Reverse reduced shape: {reverse_reduced.shape}")



# Perform MBAR analysis
mbar = MBAR(u_kn, N_k) #dimensions are off here


'''
Delta_f_ij: this is the free energy difference between two states (i and i+1) 
it tells you how favorable the transition is from one state to another
a negative Delta_f_ij means that the transition from state i to state i+1 is favorable, while a positive value means it's less favorable

dDelta_f_ij: this represents the uncertainty or standard error in the free energy difference Delta_f_ij
it gives you an estimate of how confident you can be in the computed free energy difference
a smaller dDelta_f_ij means a more precise estimate
'''

Delta_f_ij, dDelta_f_ij = mbar.compute_free_energy_differences()

print(f"Type of Delta_f_ij: {type(Delta_f_ij)}")
print(f"Type of dDelta_f_ij: {type(dDelta_f_ij)}")

print(f"Shape of Delta_f_ij: {np.shape(Delta_f_ij)}")
print(f"Shape of dDelta_f_ij: {np.shape(dDelta_f_ij)}")


print(f"Free energy difference: {Delta_f_ij[0, 1]} ± {dDelta_f_ij[0, 1]} kJ/mol")