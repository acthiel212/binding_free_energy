import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from intspan import intspan
import os

def saveState(simulation, filename):
    simulation.saveState(filename)
    print(f"State saved to {filename}")

def loadState(simulation, filename):
    simulation.loadState(filename)
    print(f"State loaded from {filename}")

# Argument parser for user-defined flags
parser = argparse.ArgumentParser(description='OpenMM Equilibration Simulation with Custom Flags')
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

# New arguments for equilibration output
parser.add_argument('--output_pdb', type=str, default='final_equil.pdb', help='Specify the output PDB filename (default: final_equil.pdb)')
parser.add_argument('--output_xml', type=str, default='final_equil.xml', help='Specify the output XML filename (default: final_equil.xml)')

args = parser.parse_args()

# Parse alchemical_atoms input
alchemical_atoms = list(intspan(args.alchemical_atoms))
# OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
alchemical_atoms = [i - 1 for i in alchemical_atoms]
print(f"Alchemical atoms: {alchemical_atoms}")

use_restraint = args.use_restraints
if use_restraint:
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

# Create the restraint force
if use_restraint:
    convert = openmm.kJ_per_kcal / (openmm.Nm_per_angstrom * openmm.Nm_per_angstrom)
    restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
    restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
    restraint.setForceGroup(0)
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("l")
    restraint.addPerBondParameter("u")
    restraint.addGroup(restraint_atoms_1)
    restraint.addGroup(restraint_atoms_2)
    restraint.addBond([0, 1], [restraint_constant * convert, restraint_lower_distance * openmm.Nm_per_angstrom,
                               restraint_upper_distance * openmm.Nm_per_angstrom])
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
required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce']
for force in required_forces:
    if force not in forceDict:
        raise ValueError(f"Required force '{force}' not found in the system.")

vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
vdwForce.setForceGroup(1)
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
multipoleForce.setForceGroup(1)

# Initialize the integrator
integrator = MTSLangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond, [(0,8),(1,1)])

# Select platform
properties = {'CUDA_Precision': 'double'}
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)
context.setParameter("AmoebaVdwLambda", vdw_lambda)
state = context.getState(getEnergy=True, getPositions=True)
print(f"Initial Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Set alchemical method and parameters for van der Waals force
vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
for i in alchemical_atoms:
    params = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, params[0], params[1], params[2], params[3], True, params[5])
# Update force parameters
vdwForce.updateParametersInContext(context)

# Apply alchemical scaling to the multipole force
for i in alchemical_atoms:
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0] * elec_lambda
    dipole = [d * elec_lambda for d in params[1]]
    quadrupole = [q * elec_lambda for q in params[2]]
    polarizability = params[-1] * elec_lambda
    # Update multipole parameters (keeping other parameters unchanged)
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
# Update force parameters
multipoleForce.updateParametersInContext(context)

# Reinitialize the context to ensure changes are applied
context.reinitialize(preserveState=True)
state = context.getState(getEnergy=True, getPositions=True)
print(f"AmoebaVdwLambda: {context.getParameter('AmoebaVdwLambda')}")
print(f"Potential Energy after reinitialization: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Perform energy minimization
print("Starting energy minimization...")
simulation.minimizeEnergy(tolerance=Quantity(value=10, unit=kilojoule / nanometer / mole), maxIterations=0)
print("Energy minimization completed.")

# Optionally, you can print the minimized energy
state = simulation.context.getState(getEnergy=True)
print(f"Minimized Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")

# Add minimal reporter (optional)
# If you want to monitor progress without excessive output, you can add a single StateDataReporter
# Here, we skip adding reporters to avoid printing every step

# Run equilibration MD steps
print(f"Starting equilibration for {nSteps} steps...")
simulation.step(nSteps)
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
