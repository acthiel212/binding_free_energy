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
parser.add_argument('--vdw_lambda_i', required=True, type=float, help='Lambda for van der Waals at state i')
parser.add_argument('--elec_lambda_i', required=True, type=float, help='Lambda for electrostatics at state i')
parser.add_argument('--vdw_lambda_ip1', required=True, type=float, help='Lambda for van der Waals at state i+1')
parser.add_argument('--elec_lambda_ip1', required=True, type=float, help='Lambda for electrostatics at state i+1')
parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')

# Restraint parameters, provided if restraints are enabled
parser.add_argument('--use_restraints', required=False, type=bool, help='Whether to use restraint', default=False)
parser.add_argument('--restraint_atoms_1', required=False, type=str, help='Range of atoms in restraint group 1 (e.g., "0,2")', default="")
parser.add_argument('--restraint_atoms_2', required=False, type=str, help='Range of atoms in restraint group 2 (e.g., "0,2")', default="")
parser.add_argument('--restraint_constant', type=float, help='Restraint force constant (default: 0)', default=0)
parser.add_argument('--restraint_lower_distance', type=float, help='Restraint lower distance (default: 0.0)', default=0.0)
parser.add_argument('--restraint_upper_distance', type=float, help='Restraint upper distance (default: 0)', default=0)

# Parse the arguments
args = parser.parse_args()


# Helper function to set lambda values and update forces in the context
def set_lambda_values(context, vdw_lambda, elec_lambda, vdwForce, multipoleForce, alchemical_atoms):
    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
    
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
    context.reinitialize(preserveState=True)

# Function to add restraints to the simulation context
def add_restraints(system, restraint_atoms_1, restraint_atoms_2, restraint_constant, restraint_lower_distance, restraint_upper_distance):
    custom_bond_force = CustomBondForce(f"step(r-{restraint_lower_distance})*step({restraint_upper_distance}-r)*0.5*{restraint_constant}*(r-{restraint_lower_distance})^2")
    custom_bond_force.addPerBondParameter('r0')
    for atom1, atom2 in zip(restraint_atoms_1, restraint_atoms_2):
        custom_bond_force.addBond(atom1, atom2, [])
    system.addForce(custom_bond_force)
    return custom_bond_force

def setup_simulation(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms):
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(forcefield_file)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*femtosecond)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)

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
    vdwForce.setAlchemicalMethod(2)
    return context, vdwForce, multipoleForce, system, pdb


def compute_work(traj_file1, traj_file2, context, pdb, vdw_lambda_1, vdw_lambda_2, elec_lambda_1, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms):
    traji = load_dcd(traj_file1, pdb_file)
    trajip1 = load_dcd(traj_file2, pdb_file)
    energy11, energy12, energy22, energy21 = [], [], [], []

    # Forward work: lambda i -> lambda i+1
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy11 = []
    for frame in range(len(traji)):
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print("energy11["+str(frame)+"]: " + str(potential_energy))
        energy11.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy12 = []
    for frame in range(len(traji)):
        context.setPositions(traji.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print("energy12["+str(frame)+"]: " + str(potential_energy))
        energy12.append(potential_energy)
    
    forward_work = np.array(energy12) - np.array(energy11)

    # Reverse work: lambda i+1 -> lambda i
    set_lambda_values(context, vdw_lambda_2, elec_lambda_2, vdwForce, multipoleForce, alchemical_atoms)
    energy22 = []
    for frame in range(len(trajip1)):
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print("energy22["+str(frame)+"]: " + str(potential_energy))
        energy22.append(potential_energy)
    
    set_lambda_values(context, vdw_lambda_1, elec_lambda_1, vdwForce, multipoleForce, alchemical_atoms)
    energy21 = []
    for frame in range(len(trajip1)):
        context.setPositions(trajip1.openmm_positions(frame))
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print("energy21["+str(frame)+"]: " + str(potential_energy))
        energy21.append(potential_energy)

    reverse_work = np.array(energy21) - np.array(energy22)

    return forward_work, reverse_work

# Main execution
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
alchemical_atoms = list(intspan(args.alchemical_atoms))
alchemical_atoms = [i - 1 for i in alchemical_atoms]
print(alchemical_atoms)
nonbonded_method = {'NoCutoff': NoCutoff, 'CutoffNonPeriodic': CutoffNonPeriodic, 'PME': PME, 'Ewald': Ewald}[args.nonbonded_method]
nonbonded_cutoff = 1.0 * nanometer

# Create restraints if necessary
if args.use_restraints:
        restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
        restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
        restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
        restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]
        create_restraint(system, args, restraint_atoms_1, restraint_atoms_2)
    

context, vdwForce, multipoleForce, system, pdb = setup_simulation(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms)

# Forward and reverse work calculation
forward_work, reverse_work = compute_work(args.traj_i, args.traj_ip1, context, pdb, args.vdw_lambda_i, args.vdw_lambda_ip1, args.elec_lambda_i, args.elec_lambda_ip1, vdwForce, multipoleForce, alchemical_atoms)

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

if(args.use_restraints is False):
    context, vdwForce, multipoleForce, system, pdb = setup_simulation(
        pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms
    )
else:
    context, vdwForce, multipoleForce, system, pdb = setup_simulation(
        pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff, alchemical_atoms, 
        use_restraint=True, restraint_atoms_1=args.restraint_atoms_1, restraint_atoms_2=args.restraint_atoms_2, 
        restraint_constant=args.restraint_constant, restraint_lower_distance=args.restraint_lower_distance, restraint_upper_distance=args.restraint_upper_distance
    )


forward_work, reverse_work = compute_work(
    args.traj_i, args.traj_ip1, context, pdb, args.vdw_lambda_i, args.vdw_lambda_ip1,
    args.elec_lambda_i, args.elec_lambda_ip1, vdwForce, multipoleForce, alchemical_atoms
)

print('Forward Work:', forward_work)
print('Reverse Work:', reverse_work)
