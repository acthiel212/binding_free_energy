from utils import File_Parser, Restart_Parser
from alchemistry import Harmonic_Restraint
from alchemistry import Alchemical
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from intspan import intspan
import os

parser = File_Parser.create_parser()
args = parser.parse_args()

# Convert nonbonded_method string to OpenMM constant
nonbonded_method_map = {
    'NoCutoff': NoCutoff,
    'CutoffNonPeriodic': CutoffNonPeriodic,
    'PME': PME,
    'Ewald': Ewald
}
nonbonded_method = nonbonded_method_map.get(args.nonbonded_method, NoCutoff)

# Load PDB and Force Field
pdb = PDBFile(args.pdb_file)
forcefield = ForceField(args.forcefield_file)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=args.nonbonded_method,
                                 nonbondedCutoff=args.nonbonded_cutoff*nanometer, constraints=None)

# Create the restraint force
if args.use_restraint:
    restraint = Harmonic_Restraint.create_restraint(args.restraint_atoms_1, args.restraint_atoms_2, args.restraint_constant,
                                                    args.restraint_lower_distance, args.restraint_upper_distance)
    system.addForce(restraint)
    print("Adding Restraint with parameters: ", restraint.getBondParameters(0))
    restraint.setUsesPeriodicBoundaryConditions(True)
    print("Using PBC Conditions on Restraint? ", restraint.usesPeriodicBoundaryConditions())

vdwForce, multipoleForce = Alchemical.setup_alchemical_forces(system)

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
state = context.getState(getEnergy=True, getPositions=True)
print(f"Initial Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# If checkpoint exists, load and restart
checkpoint_filename = Restart_Parser.get_checkpoint_filename(args.prefix)
if os.path.exists(checkpoint_filename):
    Restart_Parser.loadCheckpoint(simulation, checkpoint_filename)
# If not reinitialize the system with appropriate alchemical lambdas
else:
    context = Alchemical.apply_lambdas(context, args.alchemical_atoms, vdwForce, args.vdw_lambda,
                                       multipoleForce, args.elec_lambda)

state = context.getState(getEnergy=True, getPositions=True)
print(f"AmoebaVdwLambda: {context.getParameter('AmoebaVdwLambda')}")
print(f"AmoebaElecLambda: {args.elec_lambda}")
print(f"Potential Energy after reinitialization: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Add reporters
simulation.reporters.append(DCDReporter(args.name_dcd, 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, speed=True, separator=', '))
simulation.reporters.append(CheckpointReporter(checkpoint_filename, args.checkpoint_freq))
simulation.step(args.nSteps-simulation.currentStep)
os.makedirs(os.path.dirname(args.checkpoint_prefix), exist_ok=True) if os.path.dirname(args.checkpoint_prefix) else None
print("Simulation completed successfully.")
exit()
