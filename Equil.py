from utils import Parser_Utils, Restart_Utils
from alchemistry import Harmonic_Restraint
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import os


def parse_atom_selection(selection):
    """Parse 1-based atom ranges like '1-10,15,20-25' into 0-based indices."""
    if not selection:
        return []

    atom_indices = set()
    tokens = [token.strip() for token in selection.split(',') if token.strip()]
    for token in tokens:
        if '-' in token:
            start_str, end_str = token.split('-', 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            atom_indices.update(range(start - 1, end))
        else:
            atom_indices.add(int(token) - 1)

    return sorted(atom_indices)


def freeze_particle_masses(system, atom_selections):
    """Set masses to zero for selected atoms (OpenMM positional constraints via zero mass)."""
    frozen_indices = set()
    for selection in atom_selections:
        frozen_indices.update(parse_atom_selection(selection))

    num_particles = system.getNumParticles()
    for atom_index in frozen_indices:
        if atom_index < 0 or atom_index >= num_particles:
            raise ValueError(f"Atom index {atom_index + 1} is outside valid range 1-{num_particles}.")
        system.setParticleMass(atom_index, 0 * dalton)

    return len(frozen_indices)

parser = Parser_Utils.create_default_parser()
parser = Parser_Utils.add_dynamics_parser(parser)
parser = Parser_Utils.add_restraint_parser(parser)
parser.add_argument('--barostat_mode', required=False, type=str, choices=['equil', 'npt'], default='equil',
                    help='Set to equil to include MonteCarloBarostat, or npt to omit it for the pre-equil stage.')
parser.add_argument('--freeze_atoms_host', required=False, type=str, default='',
                    help='Optional 1-based host atom indices/ranges to freeze by setting mass to zero.')
parser.add_argument('--freeze_atoms_guest', required=False, type=str, default='',
                    help='Optional 1-based guest atom indices/ranges to freeze by setting mass to zero.')
args = parser.parse_args()

use_barostat = args.barostat_mode == 'equil'

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

frozen_count = freeze_particle_masses(system, [args.freeze_atoms_host, args.freeze_atoms_guest])
if frozen_count > 0:
    print(f"Applied zero-mass constraints to {frozen_count} atoms before {args.barostat_mode} stage.")

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

if nonbonded_method == PME and use_barostat:
    system.addForce(MonteCarloBarostat(1.0, 298.0, 25))

# Setup simulation context
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    force_name = system.getForce(i).getName()
    forceDict[force_name] = i
print(f"Force dictionary: {forceDict}")

# Ensure that the specified forces exist
required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce']
if use_barostat and nonbonded_method == PME:
    required_forces.append('MonteCarloBarostat')

for force in required_forces:
    if force not in forceDict:
        raise ValueError(f"Required force '{force}' not found in the system.")

vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
vdwForce.setForceGroup(1)
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
multipoleForce.setForceGroup(1)
if use_barostat and 'MonteCarloBarostat' in forceDict:
    barostat = system.getForce(forceDict.get('MonteCarloBarostat'))
    barostat.setForceGroup(1)
else:
    print(f"Running {args.barostat_mode} stage without MonteCarloBarostat.")

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, args.step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'mixed'}
simulation = Simulation(pdb.topology, system, integrator, None)

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
remaining_steps = max(args.num_steps - simulation.currentStep, 0)
print(f"Starting {args.barostat_mode} stage for {remaining_steps} steps...")
simulation.reporters.append(StateDataReporter(stdout, 100, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, speed=True, separator=', '))
simulation.reporters.append(CheckpointReporter(checkpoint_filename, args.checkpoint_freq, writeState=True))
if remaining_steps > 0:
    simulation.step(remaining_steps)
else:
    print("No steps remaining for this stage; skipping integration.")
os.makedirs(os.path.dirname(args.checkpoint_prefix), exist_ok=True) if os.path.dirname(args.checkpoint_prefix) else None
print(f"{args.barostat_mode.capitalize()} stage completed.")

# Save the final state to PDB
output_pdb = args.output_pdb
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
print(f"{args.barostat_mode.capitalize()} simulation completed successfully.")
