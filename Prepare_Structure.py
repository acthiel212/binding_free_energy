from utils import File_Parser
from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
from utils.File_Writer import transfer_CONECT_records
from sys import stdout

parser = File_Parser.create_default_parser()
args = parser.parse_args()

# Load PDB and Force Field
pdb = PDBFile(args.pdb_file)
forcefield = ForceField(args.forcefield_file)

# Add solvent
# Manual for PDBFixer (https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html)
fixer = PDBFixer(args.pdb_file)
fixer.addSolvent(Vec3(5, 5, 5)*nanometer)

#Write out solvated file and transfer CONECT files to solv structure
PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{args.pdb_file}_solv", 'w'))
transfer_CONECT_records(args.pdb_file, f"{args.pdb_file}_solv")

#Open new pdb with CONECT records that OpenMM requires for residues
pdb_solv = PDBFile(f"{args.pdb_file}_solv")

#Display Forces, minimize system, and write minimized structure
print('Minimizing...')
system = forcefield.createSystem(pdb_solv.topology, nonbondedMethod=PME)
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    force_name = system.getForce(i).getName()
    forceDict[force_name] = i
print(f"Force dictionary: {forceDict}")
integrator = VerletIntegrator(0.001*picoseconds)
simulation = Simulation(pdb_solv.topology, system, integrator)
simulation.context.setPositions(pdb_solv.positions)
state = simulation.context.getState(getEnergy=True, getPositions=True)
print(f"Initial Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")
simulation.minimizeEnergy(tolerance=10, maxIterations=0)
state = simulation.context.getState(getEnergy=True)
print(f"Minimized Potential Energy: {state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}")
print('Saving...')
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open(f"{args.pdb_file}_solv_min", 'w'))
print('Done')

