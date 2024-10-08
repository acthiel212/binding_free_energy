import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from intspan import intspan

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


args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = list(intspan(args.alchemical_atoms))
# OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
alchemical_atoms = [i - 1 for i in alchemical_atoms]
print(alchemical_atoms)
use_restraint = args.use_restraints
if(use_restraint):
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

# create the restraint force
if(use_restraint):
    convert = openmm.KJPerKcal / (openmm.NmPerAngstrom * openmm.NmPerAngstrom)
    restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
    restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
    restraint.setForceGroup(0)
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("l")
    restraint.addPerBondParameter("u")
    restraint.addGroup(restraint_atoms_1)
    restraint.addGroup(restraint_atoms_2)
    restraint.addBond([0, 1], [restraint_constant * convert, restraint_lower_distance * openmm.NmPerAngstrom,
                               restraint_upper_distance * openmm.NmPerAngstrom])
    system.addForce(restraint)
    print("Adding Restraint with parameters: ", restraint.getBondParameters(0))
    restraint.setUsesPeriodicBoundaryConditions(True)
    print("Using PBC Conditions on Restraint? ", restraint.usesPeriodicBoundaryConditions())


# Setup simulation context
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    forceDict[system.getForce(i).getName()] = i
print(forceDict)
vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
vdwForce.setForceGroup(1)
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
multipoleForce.setForceGroup(1)
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond, [(0,8),(1,1)])

properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
context.setParameter("AmoebaVdwLambda", vdw_lambda)
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))

# Set alchemical method and parameters for van der Waals force
vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
for i in alchemical_atoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
#update force parameters
vdwForce.updateParametersInContext(context)

# apply alchemical scaling to the multipole force
for i in alchemical_atoms:
    # adjust the unpacking based on the number of returned parameters
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0]
    dipole = params[1]
    quadrupole = params[2]
    polarizability = params[-1]
    # scale dipole and quadrupole components by electrostaticLambda
    charge = charge * elec_lambda
    dipole = [d * elec_lambda for d in dipole]
    quadrupole = [q * elec_lambda for q in quadrupole]
    polarizability = polarizability * elec_lambda
    # update multipole parameters (keeping other parameters unchanged)
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
#update force parameters
multipoleForce.updateParametersInContext(context)

# Reinitialize the context to ensure changes are applied
context.reinitialize(preserveState=True)
state = context.getState(getEnergy=True, getPositions=True)
print(context.getParameter("AmoebaVdwLambda"))
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
name_dcd = args.name_dcd
simulation.reporters.append(DCDReporter(name_dcd, 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, speed=True, separator=', '))
simulation.step(nSteps)

# to test:
# python BindingFreeEnergy.py --pdb_file "temoa_g3-15-0000-0000.pdb"  --forcefield_file "hostsG3.xml" --nonbonded_method "PME" --num_steps 15000 --step_size 2 --nonbonded_cutoff 1.0 --vdw_lambda 0 --elec_lambda 0.0 --alchemical_atoms "197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216" --restraint_atoms_1 "15,16,17,18,19,20,60,61,62,63,64,65,105,106,107,108,109,110,150,151,152,153,154,155" --restraint_atoms_2 "198,199,200,201,202" --restraint_constant 15 --restraint_lower_distance 0.0 --restraint_upper_distance 3.0 --name_dcd output0.dcd