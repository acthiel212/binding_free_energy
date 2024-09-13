import argparse
import mdtraj as md
from openmm.app import *
from openmm import *
from openmm.unit import *
from intspan import intspan
from sys import stdout
import numpy as np

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
parser.add_argument('--binding_energy_file', required=True, type=str, help='File containing binding free energy values for comparison')
args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = list(intspan(args.alchemical_atoms))

# Convert nonbonded_method string to OpenMM constant
nonbonded_method_map = {
    'NoCutoff': NoCutoff,
    'CutoffNonPeriodic': CutoffNonPeriodic,
    'PME': PME,
    'Ewald': Ewald
}
nonbonded_method = nonbonded_method_map.get(args.nonbonded_method, NoCutoff)

# Define flags based on user input
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
nSteps = args.num_steps
step_size = args.step_size * femtoseconds
nonbonded_cutoff = args.nonbonded_cutoff * nanometer
vdw_lambda = args.vdw_lambda
elec_lambda = args.elec_lambda

# Load the DCD file and PDB file
dcd_file = 'output.dcd'
traj = md.load_dcd(dcd_file, top=pdb_file)

# Create an OpenMM system based on the topology
pdb = PDBFile(pdb_file)
forcefield = ForceField(forcefield_file)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)

# Set up an integrator and platform
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, step_size)
platform = Platform.getPlatformByName('CUDA')

# Initialize the context for the simulation
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context

# Add a reporter to output the third comparison
reporter_data = []
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True, separator=', '))

# Check for van der Waals and multipole forces
vdwForce = None
multipoleForce = None
for force in system.getForces():
    if isinstance(force, AmoebaVdwForce):
        vdwForce = force
    elif isinstance(force, AmoebaMultipoleForce):
        multipoleForce = force

# Ensure van der Waals force is found
if vdwForce is None:
    raise Exception("AmoebaVdwForce not found in system.")
if multipoleForce is None:
    raise Exception("AmoebaMultipoleForce not found in system.")

# Apply alchemical scaling for van der Waals and electrostatic interactions
vdwForce.setAlchemicalMethod(2)  # Alchemical annihilation
for i in alchemical_atoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
vdwForce.updateParametersInContext(context)

for i in alchemical_atoms:
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0] * elec_lambda
    dipole = [d * elec_lambda for d in params[1]]
    quadrupole = [q * elec_lambda for q in params[2]]
    polarizability = params[-1] * elec_lambda
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
multipoleForce.updateParametersInContext(context)

# Calculate and store potential energies for each frame in the DCD file
dcd_energies = []
reporter_energies = []
for frame in range(len(traj)):
    # Set positions from the DCD frame
    simulation.context.setPositions(traj.openmm_positions(frame))
    
    # Get potential energy from DCD
    state = simulation.context.getState(getEnergy=True)
    potential_energy_dcd = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    dcd_energies.append(potential_energy_dcd)
    
    # Add reporter potential energy (simulated from StateDataReporter)
    reporter_state = simulation.context.getState(getEnergy=True)
    reporter_energy = reporter_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    reporter_energies.append(reporter_energy)

# Load energies from the provided binding energy file
binding_energies = np.loadtxt(args.binding_energy_file)

# Output energies from the DCD file, binding energy file, and reporter
for i in range(min(len(dcd_energies), len(binding_energies), len(reporter_energies))):
    dcd_energy = dcd_energies[i]
    binding_energy = binding_energies[i]
    reporter_energy = reporter_energies[i]
    print(f"Frame {i}: DCD energy = {dcd_energy}, Binding energy = {binding_energy}, Reporter energy = {reporter_energy}")

print("Energy output complete.")
