from pymbar import other_estimators
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from mdtraj import load_dcd

from alchemistry import Harmonic_Restraint, Alchemical
from utils import File_Parser


def compute_work(traj_file1, traj_file2, context, pdb_file, vdw_lambda_1, vdw_lambda_2, elec_lambda_1, elec_lambda_2,
                 vdwForce, multipoleForce, alchemical_atoms, default_elec_params):
    traji = load_dcd(traj_file1, pdb_file)
    trajip1 = load_dcd(traj_file2, pdb_file)

    # Adjust frames based on start, stop, and step_size
    start = args.start
    stop = args.stop if args.stop is not None else len(traji)
    step_size = args.step_size

    frames_to_process_i = range(start, stop, step_size)
    frames_to_process_ip1 = range(start, stop, step_size)

    energy11, energy12, energy22, energy21 = [], [], [], []

    # Forward work: lambda i -> lambda i+1
    Alchemical.update_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms,
                                    default_elec_params)
    energy11 = []
    for frame in frames_to_process_i:
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy11.append(potential_energy)

    Alchemical.update_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms,
                                    default_elec_params)
    energy12 = []
    for frame in frames_to_process_i:
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy12.append(potential_energy)

    forward_work = np.array(energy12) - np.array(energy11)

    # Reverse work: lambda i+1 -> lambda i
    Alchemical.update_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms,
                                    default_elec_params)
    energy22 = []
    for frame in frames_to_process_ip1:
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy22.append(potential_energy)

    Alchemical.update_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms,
                                    default_elec_params)
    energy21 = []
    for frame in frames_to_process_ip1:
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy21.append(potential_energy)

    reverse_work = np.array(energy21) - np.array(energy22)

    return forward_work, reverse_work


# Main execution
# Argument parser for user-defined flags
parser = File_Parser.create_default_parser()
parser = File_Parser.add_restraint_parser(parser)
parser = File_Parser.add_BAR_parser(parser)
args = parser.parse_args()

# Load PDB and Force Field
pdb = PDBFile(args.pdb_file)
forcefield = ForceField(args.forcefield_file[0])
if (len(args.forcefield_file) > 1):
    for file in args.forcefield_file[1:]:
        forcefield.loadFile(file)


# Convert nonbonded_method string to OpenMM constant
nonbonded_method_map = {
    'NoCutoff': NoCutoff,
    'CutoffNonPeriodic': CutoffNonPeriodic,
    'PME': PME,
    'Ewald': Ewald
}
nonbonded_method = nonbonded_method_map.get(args.nonbonded_method, NoCutoff)
nonbonded_cutoff = args.nonbonded_cutoff * nanometer
step_size = args.step_size
system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff,
                                 constraints=None)

# Create the restraint force
if args.use_restraints:
    restraint = Harmonic_Restraint.create_restraint(args.restraint_atoms_1, args.restraint_atoms_2,
                                                    args.restraint_constant,
                                                    args.restraint_lower_distance, args.restraint_upper_distance)
    system.addForce(restraint)
    print("Adding Restraint with parameters: ", restraint.getBondParameters(0))
    restraint.setUsesPeriodicBoundaryConditions(True)
    print("Using PBC Conditions on Restraint? ", restraint.usesPeriodicBoundaryConditions())

# Setup simulation context
numForces = system.getNumForces()
integrator = MTSLangevinIntegrator(300 * kelvin, 1 / picosecond, step_size * femtosecond, [(0, 8), (1, 1)])
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context

vdwForce, multipoleForce = Alchemical.setup_alchemical_forces(system)
default_elec_params = Alchemical.save_default_elec_params(multipoleForce, args.alchemical_atoms)

# Forward and reverse work calculation
forward_work, reverse_work = compute_work(args.traj_i, args.traj_ip1, context, args.pdb_file, args.vdw_lambda_i,
                                          args.vdw_lambda_ip1, args.elec_lambda_i, args.elec_lambda_ip1, vdwForce,
                                          multipoleForce, args.alchemical_atoms, default_elec_params)

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

# TODO: create a num_steps and a step_size flag for traversing the dcd file and also create a start flag and a stop flag in case the user wants to start or stop at a specific snapshot
# to test:
# python BAR.py --traj_i output0.dcd --traj_ip1 output1.dcd --pdb_file "temoa_g3-15-0000-0000.pdb" --forcefield_file "hostsG3.xml" --nonbonded_method "PME" --num_steps 15000 --step_size 2 --nonbonded_cutoff 1.0 --vdw_lambda_i 0 --elec_lambda_i 0 --vdw_lambda_ip1 0.4 --elec_lambda_ip1 0 --alchemical_atoms "197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216" --restraint_atoms_1 "15,16,17,18,19,20,60,61,62,63,64,65,105,106,107,108,109,110,150,151,152,153,154,155" --restraint_atoms_2 "198,199,200,201,202" --restraint_constant 15 --restraint_lower_distance 0.0 --restraint_upper_distance 3.0
