#BindingFreeEnergy.py
import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from intspan import intspan
from sys import stdout

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

args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = list(intspan(args.alchemical_atoms))
use_restraint = args.use_restraints
if(use_restraint):
    restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
    restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
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

# Create the restraint force
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
    restraint.addBond([0, 1], [restraint_constant, restraint_lower_distance, restraint_upper_distance])
    system.addForce(restraint)

# Setup simulation context
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    forceDict[system.getForce(i).getName()] = i
print(forceDict)
vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond)

properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')

# This avoids reusing the same integrator that is already bound to the previous context.
new_integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond)

# Create a new simulation context
simulation = Simulation(pdb.topology, system, new_integrator, platform)
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
# Update force parameters
vdwForce.updateParametersInContext(context)

# Apply alchemical scaling to the multipole force
for i in alchemical_atoms:
    # Adjust the unpacking based on the number of returned parameters
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0]
    dipole = params[1]
    quadrupole = params[2]
    polarizability = params[-1]
    # Scale dipole and quadrupole components by electrostaticLambda
    charge = charge * elec_lambda
    dipole = [d * elec_lambda for d in dipole]
    quadrupole = [q * elec_lambda for q in quadrupole]
    polarizability = polarizability * elec_lambda
    # Update multipole parameters (keeping other parameters unchanged)
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
# Update force parameters
multipoleForce.updateParametersInContext(context)

# Reinitialize the context to ensure changes are applied
context.reinitialize(preserveState=True)
state = context.getState(getEnergy=True, getPositions=True)
print(context.getParameter("AmoebaVdwLambda"))
# Open the file in append mode
with open('binding_energies.txt', 'w') as f:
    # Add reporters
    simulation.reporters.append(DCDReporter('output2.dcd', 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=', ', elapsedTime=True))

    # Run the simulation step by step
    for step in range(1, nSteps+1):
        simulation.step(1)

        # Only log the energy to the file every 1000 steps (to match console output)
        if (step % 1000 == 0):
            # Get the state of the system after this step
            state = simulation.context.getState(getEnergy=True)
            potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            f.write(f"{potential_energy}\n")

            



# Simulation complete
print("Simulation complete and energies saved to binding_energies.txt.")

