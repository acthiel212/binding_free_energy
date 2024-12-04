from utils import Parser_Utils
from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
from utils.File_Conversion_Utils import transfer_CONECT_records
from sys import stdout

parser = Parser_Utils.create_default_parser()
args = parser.parse_args()

# Load PDB and Force Field
pdb = PDBFile(args.pdb_file)
filename=os.path.splitext(args.pdb_file)[0]
forcefield = ForceField(args.forcefield_file[0])
if (len(args.forcefield_file) > 1):
    for file in args.forcefield_file[1:]:
        forcefield.loadFile(file)

# Add solvent
# Manual for PDBFixer (https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html)
fixer = PDBFixer(args.pdb_file)
fixer.addSolvent(Vec3(5, 5, 5)*nanometer)

#Write out solvated file and transfer CONECT files to solv structure
PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{filename}_solv.pdb", 'w'))
transfer_CONECT_records(args.pdb_file, f"{filename}_solv.pdb")

#Open new pdb with CONECT records that OpenMM requires for residues
pdb_solv = PDBFile(f"{filename}_solv.pdb")

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
PDBFile.writeFile(simulation.topology, positions, open(f"{filename}_solv_min.pdb", 'w'))
print('Done')

