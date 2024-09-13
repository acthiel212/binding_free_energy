import mdtraj as md
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from intspan import intspan

# Define the DCD file, topology, and parameters
dcd_file = 'output.dcd'

pdb_file = 'g2.pdb'
forcefield_file = 'g1.xml'
nSnapshots = 30  # number of snapshots to compare
vdwLambda = 1
elecLambda = 1
alchemicalAtoms = list(intspan("0,2"))

# Load the DCD file
traj = md.load_dcd(dcd_file, top=pdb_file)
print(f"Loaded {len(traj)} frames from {dcd_file}.")

# Load the system from PDB and force field
pdb = PDBFile(pdb_file)
forcefield = ForceField(forcefield_file)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)

# Check force types and identify van der Waals and multipole forces
for i, force in enumerate(system.getForces()):
    print(f"Force {i}: {type(force)}")

# Assuming we know that AmoebaVdwForce and AmoebaMultipoleForce are at specific indices
vdwForce = None
multipoleForce = None

# Find the AmoebaVdwForce and AmoebaMultipoleForce
for force in system.getForces():
    if isinstance(force, AmoebaVdwForce):
        vdwForce = force
    elif isinstance(force, AmoebaMultipoleForce):
        multipoleForce = force

# Ensure we have the correct forces
if vdwForce is None:
    raise Exception("AmoebaVdwForce not found in system.")
if multipoleForce is None:
    raise Exception("AmoebaMultipoleForce not found in system.")

# Apply alchemical method to the van der Waals force
vdwForce.setAlchemicalMethod(2)  # Choose correct alchemical method

# Modify parameters for alchemical atoms
for i in alchemicalAtoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)

# Set up the context for simulation
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)

# You must first create the Context before updating parameters
context = simulation.context

# Update the van der Waals force parameters in the context
vdwForce.updateParametersInContext(context)

# Initialize the simulation context and compute energies for each frame
dcd_energies = []

for frame in range(min(nSnapshots, len(traj))):
    # Set positions from the DCD frame
    simulation.context.setPositions(traj.openmm_positions(frame))
    
    # Get energy
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    dcd_energies.append(potential_energy)

# Load energies from BindingFreeEnergy.py output
binding_energy_file = 'binding_energies.txt'  # Assuming you've saved the energies from BindingFreeEnergy.py to a file
binding_energies = np.loadtxt(binding_energy_file)

# Compare energies (up to first decimal place)
for i in range(len(dcd_energies)):
    dcd_energy = dcd_energies[i]  # Extract the numerical value
    binding_energy = binding_energies[i]  # Assuming binding_energies are already in kcal/mol and are just numbers
    if dcd_energy != binding_energy:
        print(f"Mismatch at frame {i}: DCD energy = {dcd_energy}, Binding energy = {binding_energy}")
    else:
        print(f"Match at frame {i}: {dcd_energy} kcal/mol")


print("Energy comparison complete.")
