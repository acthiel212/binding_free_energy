from utils import File_Parser, Restart_Parser
from alchemistry import Harmonic_Restraint
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import os

parser = File_Parser.create_default_parser()
parser = File_Parser.add_dynamics_parser(parser)
parser = File_Parser.add_restraint_parser(parser)
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

if(nonbonded_method == PME):
    system.addForce(MonteCarloBarostat(1.0, 298.0, 25))
# Setup simulation context
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    force_name = system.getForce(i).getName()
    forceDict[force_name] = i
print(f"Force dictionary: {forceDict}")

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, args.step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)

# If checkpoint exists, load and restart
checkpoint_filename = Restart_Parser.get_checkpoint_filename(args.checkpoint_prefix)
if os.path.exists(checkpoint_filename):
    Restart_Parser.loadCheckpoint(simulation, checkpoint_filename)
    simulation.reporters.append(DCDReporter(args.name_dcd, 1000, append=True))
# If not minimize the system before equilibration
else:
    context = simulation.context
    context.setPositions(pdb.getPositions())
    context.setVelocitiesToTemperature(300 * kelvin)
    state = context.getState(getEnergy=True, getPositions=True)
    print(f"Initial Potential Energy: {state.getPotentialEnergy()}")
    simulation.reporters.append(DCDReporter(args.name_dcd, 1000))
    # Perform energy minimization
    print("Starting energy minimization...")
    simulation.minimizeEnergy(tolerance=Quantity(value=10, unit=kilojoule / nanometer / mole), maxIterations=0)
    print("Energy minimization completed.")

    # Optionally, you can print the minimized energy
    state = simulation.context.getState(getEnergy=True)
    print(f"Minimized Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Run equilibration MD steps
print(f"Starting equilibration for {args.num_steps-simulation.currentStep} steps...")
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, speed=True, separator=', '))
simulation.reporters.append(CheckpointReporter(checkpoint_filename, args.checkpoint_freq, writeState=True))
simulation.step(args.num_steps-simulation.currentStep)
os.makedirs(os.path.dirname(args.checkpoint_prefix), exist_ok=True) if os.path.dirname(args.checkpoint_prefix) else None
print("Equilibration completed.")

# Save the final state to PDB
output_pdb = args.output_pdb
with open(output_pdb, 'w') as pdb_out:
    PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), pdb_out)
print(f"Final PDB saved to {output_pdb}")

# Save the final system state to XML
output_xml = args.output_xml
with open(output_xml, 'w') as xml_out:
    xml_out.write(XmlSerializer.serialize(system))
print(f"Final system XML saved to {output_xml}")
print("Equilibration simulation completed successfully.")
