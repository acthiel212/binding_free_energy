import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from intspan import intspan
import os

def loadCheckpoint(simulation, filename):

    simulation.loadCheckpoint(filename)
    print(f"States loaded from {filename}")

# Function to generate checkpoint filename
def get_checkpoint_filename(prefix):
    return f"{prefix}.chk"

# Argument parser for user-defined flags
parser = argparse.ArgumentParser(description='OpenMM Alchemical Simulation with Custom Flags')
parser.add_argument('--pdb_file', required=True, type=str, help='PDB file for the simulation')
parser.add_argument('--forcefield_file', required=True, type=str, help='Force field XML file')
parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
parser.add_argument('--num_steps', required=False, type=int, help='Number of MD steps to take.', default=1000)
parser.add_argument('--step_size', required=False, type=int, help='Step size given to integrator in fs.', default=1)
parser.add_argument('--nonbonded_cutoff', required=False, type=float, help='Nonbonded cutoff in nm (default: 1.0 nm)', default=1.0)
parser.add_argument('--vdw_lambda', required=False, type=float, help='Value for van der Waals lambda (default: 1.0)', default=1.0)
parser.add_argument('--elec_lambda', required=False, type=float, help='Value for electrostatic lambda (default: 0.0)', default=0.0)
parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
parser.add_argument('--use_restraints', required=False, type=bool, help='Whether to use restraint', default=False)
parser.add_argument('--restraint_atoms_1', required=False, type=str, help='Range of atoms in restraint group 1 (e.g., "0,2")', default="")
parser.add_argument('--restraint_atoms_2', required=False, type=str, help='Range of atoms in restraint group 2 (e.g., "0,2")', default="")
parser.add_argument('--restraint_constant', required=False, type=float, help='Restraint force constant (default: 1.0)', default=1.0)
parser.add_argument('--restraint_lower_distance', required=False, type=float, help='Restraint lower distance (default: 0.0)', default=0.0)
parser.add_argument('--restraint_upper_distance', required=False, type=float, help='Restraint upper distance (default: 1.0)', default=1.0)

parser.add_argument('--name_dcd', type=str, default='output.dcd', help='Specify the output DCD filename (default: output.dcd)')

parser.add_argument('--checkpoint_freq', type=int, default=1000, help='Frequency (in steps) to save checkpoints (default: 1000)')
parser.add_argument('--checkpoint_prefix', type=str, default='checkpoint', help='Prefix for checkpoint filenames (default: checkpoint)')

args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = list(intspan(args.alchemical_atoms))
# OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
alchemical_atoms = [i - 1 for i in alchemical_atoms]
print(f"Alchemical atoms: {alchemical_atoms}")

use_restraint = args.use_restraints
if use_restraint:
    restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
    restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
    # OpenMM atom index starts at zero while FFX starts at 1.
    restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
    restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]
    restraint_constant = args.restraint_constant
    restraint_lower_distance = args.restraint_lower_distance
    restraint_upper_distance = args.restraint_upper_distance

# Define flags based on user input
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
nSteps = args.num_steps
step_size = args.step_size
nonbonded_cutoff = args.nonbonded_cutoff * nanometer
vdw_lambda = args.vdw_lambda
elec_lambda = args.elec_lambda

# Initialize checkpoint parameters
checkpoint_freq = args.checkpoint_freq
checkpoint_prefix = args.checkpoint_prefix
checkpoint_filename = get_checkpoint_filename(checkpoint_prefix)
name_dcd = args.name_dcd

# Convert nonbonded_method string to OpenMM constant
nonbonded_method_map = {
    'NoCutoff': NoCutoff,
    'CutoffNonPeriodic': CutoffNonPeriodic,
    'PME': PME,
    'Ewald': Ewald
}
nonbonded_method = nonbonded_method_map.get(args.nonbonded_method, NoCutoff)

# Load PDB and Force Field
pdb = PDBFile(pdb_file)
forcefield = ForceField(forcefield_file)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)

# Create the restraint force
if use_restraint:
    convert = openmm.kJ_per_kcal / (openmm.Nm_per_angstrom * openmm.Nm_per_angstrom)
    restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2 + step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
    restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
    restraint.setForceGroup(0)
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("l")
    restraint.addPerBondParameter("u")
    restraint.addGroup(restraint_atoms_1)
    restraint.addGroup(restraint_atoms_2)
    restraint.addBond([0, 1], [restraint_constant * convert, restraint_lower_distance * openmm.Nm_per_angstrom,
                               restraint_upper_distance * openmm.Nm_per_angstrom])
    system.addForce(restraint)

# Setup simulation context
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    force_name = system.getForce(i).getName()
    forceDict[force_name] = i
print(f"Force dictionary: {forceDict}")

# Ensure that the specified forces exist
required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce']
for force in required_forces:
    if force not in forceDict:
        raise ValueError(f"Required force '{force}' not found in the system.")

vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
vdwForce.setForceGroup(1)
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
multipoleForce.setForceGroup(1)

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
if os.path.exists(checkpoint_filename):
    loadCheckpoint(simulation, checkpoint_filename)
    simulation.reporters.append(DCDReporter(name_dcd, 1000, append=True))
    simulation.reporters.append(
        StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True,
                          temperature=True, speed=True, separator=', '))
    simulation.reporters.append(CheckpointReporter(checkpoint_filename, checkpoint_freq))
    simulation.step(nSteps-simulation.currentStep)
    print("Simulation completed successfully.")
    exit()
    
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
context.setParameter("AmoebaVdwLambda", vdw_lambda)
state = context.getState(getEnergy=True, getPositions=True)
print(f"Initial Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Set alchemical method and parameters for van der Waals force
vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
for i in alchemical_atoms:
    params = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, params[0], params[1], params[2], params[3], True, params[5])
# Update force parameters
vdwForce.updateParametersInContext(context)

# Apply alchemical scaling to the multipole force
for i in alchemical_atoms:
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0] * elec_lambda
    dipole = [d * elec_lambda for d in params[1]]
    quadrupole = [q * elec_lambda for q in params[2]]
    polarizability = params[-1] * elec_lambda
    # Update multipole parameters (keeping other parameters unchanged)
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
# Update force parameters
multipoleForce.updateParametersInContext(context)

# Reinitialize the context to ensure changes are applied
context.reinitialize(preserveState=True)
state = context.getState(getEnergy=True, getPositions=True)
print(f"AmoebaVdwLambda: {context.getParameter('AmoebaVdwLambda')}")
print(f"Potential Energy after reinitialization: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Add reporters
simulation.reporters.append(DCDReporter(name_dcd, 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, speed=True, separator=', '))
simulation.reporters.append(CheckpointReporter(checkpoint_filename, checkpoint_freq))
simulation.step(nSteps)
os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True) if os.path.dirname(checkpoint_prefix) else None

print("Simulation completed successfully.")
