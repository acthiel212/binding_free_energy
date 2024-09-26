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
parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
parser.add_argument('--nonbonded_cutoff', required=False, type=float, help='Nonbonded cutoff in nm (default: 1.0 nm)', default=1.0)
parser.add_argument('--vdw_lambda_i', required=True, type=float, help='Lambda for van der Waals at state i')
parser.add_argument('--elec_lambda_i', required=True, type=float, help='Lambda for electrostatics at state i')
parser.add_argument('--vdw_lambda_ip1', required=True, type=float, help='Lambda for van der Waals at state i+1')
parser.add_argument('--elec_lambda_ip1', required=True, type=float, help='Lambda for electrostatics at state i+1')
parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')

# Restraint parameters, provided if restraints are enabled
parser.add_argument('--use_restraints', required=False, type=bool, help='Whether to use restraint', default=False)
parser.add_argument('--restraint_atoms_1', required=False, type=str, help='Range of atoms in restraint group 1 (e.g., "0,2")', default="")
parser.add_argument('--restraint_atoms_2', required=False, type=str, help='Range of atoms in restraint group 2 (e.g., "0,2")', default="")
parser.add_argument('--restraint_constant', type=float, help='Restraint force constant (default: 0)', default=0)
parser.add_argument('--restraint_lower_distance', type=float, help='Restraint lower distance (default: 0.0)', default=0.0)
parser.add_argument('--restraint_upper_distance', type=float, help='Restraint upper distance (default: 0)', default=0)

# New flags for traversing the DCD file
parser.add_argument('--step_size', type=int, required=False, help='Step size to traverse the DCD file', default=1)
parser.add_argument('--start', type=int, required=False, help='Start frame for DCD traversal', default=0)
parser.add_argument('--stop', type=int, required=False, help='Stop frame for DCD traversal', default=None)


# Parse the arguments
args = parser.parse_args()


# Helper function to set lambda values and update forces in the context
def set_lambda_values(context, vdw_lambda, elec_lambda, vdwForce, multipoleForce, alchemical_atoms, default_elec_params):
    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)

    j = 0
    for i in alchemical_atoms:
        param = default_elec_params[j]
        charge, dipole, quadrupole, polarizability = param[0], param[1], param[2], param[-1]
        charge = charge * elec_lambda
        dipole = [d * elec_lambda for d in dipole]
        quadrupole = [q * elec_lambda for q in quadrupole]
        polarizability = polarizability * elec_lambda
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *default_elec_params[j][3:-1], polarizability)
        j += 1
    vdwForce.updateParametersInContext(context)
    multipoleForce.updateParametersInContext(context)
    context.reinitialize(preserveState=True)

def save_default_elec_params(multipoleForce, alchemical_atoms):
    params = []
    for i in alchemical_atoms:
        param = multipoleForce.getMultipoleParameters(i)
        params.append(param)
    return params


def setup_simulation(pdb_file, forcefield_file, alchemical_atoms, nonbonded_method, nonbonded_cutoff=args.nonbonded_cutoff, use_restraints=args.use_restraints, 
                      restraint_atoms_1=args.restraint_atoms_1, 
                     restraint_atoms_2=args.restraint_atoms_2, restraint_constant=args.restraint_constant, 
                     restraint_lower_distance=args.restraint_lower_distance, restraint_upper_distance=args.restraint_upper_distance):
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(forcefield_file)
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
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)
    # create the restraint force
    if(use_restraints):
        convert = openmm.KJPerKcal / (openmm.NmPerAngstrom * openmm.NmPerAngstrom)
        restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
        restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
        restraint.setForceGroup(0)
        restraint.addPerBondParameter("k")
        restraint.addPerBondParameter("l")
        restraint.addPerBondParameter("u")
        restraint.addGroup(restraint_atoms_1)
        restraint.addGroup(restraint_atoms_2)
        restraint.addBond([0, 1], [restraint_constant, restraint_lower_distance, restraint_upper_distance])
        system.addForce(restraint)


    # Setup simulation context
    numForces = system.getNumForces()
    integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond, [(0,8),(1,1)])

    properties = {'CUDA_Precision': 'double'}
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(pdb.topology, system, integrator, platform)
    context = simulation.context

    vdwForce, multipoleForce = None, None
    for i in range(numForces):
        force = system.getForce(i)
        if isinstance(force, AmoebaVdwForce):
            vdwForce = force
        elif isinstance(force, AmoebaMultipoleForce):
            multipoleForce = force

    if vdwForce is None or multipoleForce is None:
        raise ValueError("AmoebaVdwForce or AmoebaMultipoleForce not found in the system.")
    vdwForce.setAlchemicalMethod(2)
    default_elec_params = save_default_elec_params(multipoleForce, alchemical_atoms)
    return context, vdwForce, multipoleForce, system, pdb, default_elec_params


def compute_work(traj_file1, traj_file2, context, pdb, vdw_lambda_1, vdw_lambda_2, elec_lambda_1, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms, default_elec_params):
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
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms, default_elec_params)
    energy11 = []
    for frame in frames_to_process_i:
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy11.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms, default_elec_params)
    energy12 = []
    for frame in frames_to_process_i:
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy12.append(potential_energy)
    
    forward_work = np.array(energy12) - np.array(energy11)

    # Reverse work: lambda i+1 -> lambda i
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms, default_elec_params)
    energy22 = []
    for frame in frames_to_process_ip1:
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy22.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms, default_elec_params)
    energy21 = []
    for frame in frames_to_process_ip1:
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy21.append(potential_energy)

    reverse_work = np.array(energy21) - np.array(energy22)

    return forward_work, reverse_work

# Main execution
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
alchemical_atoms = list(intspan(args.alchemical_atoms))
alchemical_atoms = [i - 1 for i in alchemical_atoms]
use_restraints = args.use_restraints
if(use_restraints):
    restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
    restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
    # OpenMM atom index starts at zero while FFX starts at 1.
    restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
    restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]
    restraint_constant = args.restraint_constant
    restraint_lower_distance = args.restraint_lower_distance
    restraint_upper_distance = args.restraint_upper_distance
print(alchemical_atoms)


    
    

context, vdwForce, multipoleForce, system, pdb, default_elec_params = setup_simulation(pdb_file, forcefield_file, alchemical_atoms, nonbonded_cutoff=args.nonbonded_cutoff, use_restraints=args.use_restraints,
            nonbonded_method=args.nonbonded_method, restraint_atoms_1=args.restraint_atoms_1, 
            restraint_atoms_2=args.restraint_atoms_2, restraint_constant=args.restraint_constant, 
            restraint_lower_distance=args.restraint_lower_distance, restraint_upper_distance=args.restraint_upper_distance)

# Forward and reverse work calculation
forward_work, reverse_work = compute_work(args.traj_i, args.traj_ip1, context, pdb, args.vdw_lambda_i, args.vdw_lambda_ip1, args.elec_lambda_i, args.elec_lambda_ip1, vdwForce, multipoleForce, alchemical_atoms, default_elec_params)

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


#TODO: create a num_steps and a step_size flag for traversing the dcd file and also create a start flag and a stop flag in case the user wants to start or stop at a specific snapshot
#to test:
# python BAR.py --traj_i output0.dcd --traj_ip1 output1.dcd --pdb_file "temoa_g3-15-0000-0000.pdb" --forcefield_file "hostsG3.xml" --nonbonded_method "PME" --num_steps 15000 --step_size 2 --nonbonded_cutoff 1.0 --vdw_lambda_i 0 --elec_lambda_i 0 --vdw_lambda_ip1 0.4 --elec_lambda_ip1 0 --alchemical_atoms "197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216" --restraint_atoms_1 "15,16,17,18,19,20,60,61,62,63,64,65,105,106,107,108,109,110,150,151,152,153,154,155" --restraint_atoms_2 "198,199,200,201,202" --restraint_constant 15 --restraint_lower_distance 0.0 --restraint_upper_distance 3.0 