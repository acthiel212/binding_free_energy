from utils import Parser_Utils
from alchemistry import Harmonic_Restraint
from alchemistry import Alchemical
from openmm.app import *
from openmm import *
from openmm.unit import *

# Argument parser for user-defined flags
parser = Parser_Utils.create_default_parser()
parser = Parser_Utils.add_alchemical_parser(parser)
parser = Parser_Utils.add_restraint_parser(parser)
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
forcefield = ForceField(args.forcefield_file[0])
if (len(args.forcefield_file) > 1):
    for file in args.forcefield_file[1:]:
        forcefield.loadFile(file)


system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method,
                                 nonbondedCutoff=args.nonbonded_cutoff*nanometer, constraints=None)
# Create the restraint force
if args.use_restraints:
    restraint = Harmonic_Restraint.create_restraint(args.restraint_atoms_1, args.restraint_atoms_2, args.restraint_constant,
                                                    args.restraint_lower_distance, args.restraint_upper_distance)
    system.addForce(restraint)
    print("Adding Restraint with parameters: ", restraint.getBondParameters(0))
    restraint.setUsesPeriodicBoundaryConditions(True)
    print("Using PBC Conditions on Restraint? ", restraint.usesPeriodicBoundaryConditions())


# Setup simulation context
numForces = system.getNumForces()
forceDict = {}

# Setup alchemical forces
vdwForce, multipoleForce = Alchemical.setup_alchemical_forces(system)

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
    forceDict[system.getForce(i).getName()] = i
print(forceDict)

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, 1.0*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))

# Set alchemical method and parameters for van der Waals force
Alchemical.apply_lambdas(context, args.alchemical_atoms, vdwForce, args.vdw_lambda,
                         multipoleForce, args.elec_lambda)

# Reinitialize the context to ensure changes are applied
state = context.getState(getEnergy=True, getPositions=True)
print(context.getParameter("AmoebaVdwLambda"))
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
