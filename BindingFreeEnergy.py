import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

# Argument parser for user-defined flags
parser = argparse.ArgumentParser(description='OpenMM Alchemical Simulation with Custom Flags')
parser.add_argument('--pdb_file', required=True, type=str, help='PDB file for the simulation')
parser.add_argument('--forcefield_file', required=True, type=str, help='Force field XML file')
parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
parser.add_argument('--nonbonded_cutoff', required=False, type=float, help='Nonbonded cutoff in nm (default: 1.0 nm)', default=1.0)
parser.add_argument('--vdw_lambda', required=False, type=float, help='Value for van der Waals lambda (default: 1.0)', default=1.0)
parser.add_argument('--elecLambda', required=False, type=float, help='Value for electrostatic lambda (default: 0.0)', default=0.0)
parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
parser.add_argument('--restraint_atoms_1', required=False, type=str, help='Range of atoms in restraint group 1 (e.g., "0,2")')
parser.add_argument('--restraint_atoms_2', required=False, type=str, help='Range of atoms in restraint group 2 (e.g., "0,2")')
parser.add_argument('--restraint_constant', required=False, type=float, help='Restraint force constant (default: 0.0)', default=0.0)
parser.add_argument('--restraint_lower_distance', required=False, type=float, help='Restraint lower distance (default: 0.0)', default=0.0)
parser.add_argument('--restraint_upper_distance', required=False, type=float, help='Restraint upper distance (default: 1.0)', default=1.0)

args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = range(*map(int, args.alchemical_atoms.split(',')))
restraint_atoms_1 = range(*map(int, args.restraint_atoms_1.split(',')))
restraint_atoms_2 = range(*map(int, args.restraint_atoms_2.split(',')))

# Define flags based on user input
pdb_file = args.pdb_file
forcefield_file = args.forcefield_file
nonbonded_cutoff = args.nonbonded_cutoff * nanometer
vdw_lambda = args.vdw_lambda
restraint_constant = args.restraint_constant
restraint_lower_distance = args.restraint_lower_distance
restraint_upper_distance = args.restraint_upper_distance

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
forceDict = {}
for i in range(numForces):
    forceDict[system.getForce(i).getName()] = i
print(forceDict)
vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
system.removeForce(forceDict.get('AmoebaMultipoleForce'))

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 1*femtosecond)
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
context = Context(system, integrator, platform)
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
context.setParameter("AmoebaVdwLambda", vdw_lambda)
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))

# Set alchemical method and parameters for van der Waals force
vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
for atomi in alchemical_atoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(atomi)
    vdwForce.setParticleParameters(atomi, parent, sigma, eps, redFactor, True, type)

vdwForce.updateParametersInContext(context)
context.reinitialize(preserveState=True)
print(context.getParameter("AmoebaVdwLambda"))
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
