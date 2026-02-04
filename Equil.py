from utils import Parser_Utils, Restart_Utils
from alchemistry import Harmonic_Restraint
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import os

parser = Parser_Utils.create_default_parser()
parser = Parser_Utils.add_dynamics_parser(parser)
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
    if args.restraint_type == "BORESCH":
        harmonicforce, angleforce, torsionforce = Harmonic_Restraint.create_Boresch_restraint(args.restraint_atoms_1, args.restraint_atoms_2, args.restraint_constant,
                                                                                              args.restraint_lower_distance, args.restraint_upper_distance)

        system.addForce(harmonicforce)
        system.addForce(angleforce)
        system.addForce(torsionforce)
        print("Adding Bond Restraint with parameters: ", harmonicforce.getBondParameters(0))
        print("Adding Angle Restraint with parameters: ", angleforce.getAngleParameters(0))
        print("Adding Angle Restraint with parameters: ", angleforce.getAngleParameters(1))
        print("Adding Torsion Restraint with parameters: ", torsionforce.getTorsionParameters(0))
        print("Adding Torsion Restraint with parameters: ", torsionforce.getTorsionParameters(1))
        print("Adding Torsion Restraint with parameters: ", torsionforce.getTorsionParameters(2))
        harmonicforce.setUsesPeriodicBoundaryConditions(True)
        angleforce.setUsesPeriodicBoundaryConditions(True)
        torsionforce.setUsesPeriodicBoundaryConditions(True)
        print("Using PBC Conditions on Restraint? ", harmonicforce.usesPeriodicBoundaryConditions())
    else:
        restraint = Harmonic_Restraint.create_COM_restraint(args.restraint_atoms_1, args.restraint_atoms_2,
                                                            args.restraint_constant,
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

# Ensure that the specified forces exist
required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce', 'MonteCarloBarostat']
for force in required_forces:
    if force not in forceDict:
        raise ValueError(f"Required force '{force}' not found in the system.")

vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
vdwForce.setForceGroup(1)
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
multipoleForce.setForceGroup(1)
barostat = system.getForce(forceDict.get('MonteCarloBarostat'))
barostat.setForceGroup(1)

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, args.step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'mixed'}
platform = Platform.getPlatformByName('OpenCL')
simulation = Simulation(pdb.topology, system, integrator, platform)

# If checkpoint exists, load and restart
checkpoint_filename = Restart_Utils.get_checkpoint_filename(args.checkpoint_prefix)
if os.path.exists(checkpoint_filename):
    Restart_Utils.loadCheckpoint(simulation, checkpoint_filename)
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
output_pdb = "output.pdb"
with open(output_pdb, 'w') as pdb_out:
    positions = simulation.context.getState(getPositions=True).getPositions()
    box_vectors = simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    PDBFile.writeFile(simulation.topology, positions, pdb_out)
print(f"Final PDB saved to {output_pdb}")

# Save the final system state to XML
output_xml = args.output_xml
with open(output_xml, 'w') as xml_out:
    xml_out.write(XmlSerializer.serialize(system))
print(f"Final system XML saved to {output_xml}")
print("Equilibration simulation completed successfully.")
