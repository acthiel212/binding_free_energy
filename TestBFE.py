from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
# Make these flags to the python script
pdb = PDBFile('g1.pdb_2')
forcefield = ForceField('g1.xml')
vdwLambda = 1.0
electrostaticLambda = 0.25  # example lambda for scaling multipoles
nSteps = 15000000
alchemicalAtoms = range(0, 21)
###

# create the system using the AMOEBA force field, specifying no cutoff for nonbonded interactions
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=10*nanometer, constraints=None)

# create a dictionary to map force names to their indices
numForces = system.getNumForces()
forceDict = {}
for i in range(numForces):
    forceDict[system.getForce(i).getName()] = i
print(forceDict)
# access the van der Waals and multipole forces from the system
vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))

# set up integrator and platform
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 1*femtosecond)
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
context = simulation.context
context.setPositions(pdb.getPositions())
context.setVelocitiesToTemperature(300*kelvin)

# set AmoebaVdwLambda parameter for van der Waals scaling
context.setParameter("AmoebaVdwLambda", vdwLambda)
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
# apply alchemical scaling to the van der Waals force
vdwForce.setAlchemicalMethod(2)
for i in alchemicalAtoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
    vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
#update force parameters
vdwForce.updateParametersInContext(context)

# apply alchemical scaling to the multipole force
for i in alchemicalAtoms:
    # adjust the unpacking based on the number of returned parameters
    params = multipoleForce.getMultipoleParameters(i)
    charge = params[0]
    dipole = params[1]
    quadrupole = params[2]
    polarizability = params[-1]
    # scale dipole and quadrupole components by electrostaticLambda
    charge = charge * electrostaticLambda
    dipole = [d * electrostaticLambda for d in dipole]
    quadrupole = [q * electrostaticLambda for q in quadrupole]
    polarizability = polarizability * electrostaticLambda
    # update multipole parameters (keeping other parameters unchanged)
    multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
#update force parameters
multipoleForce.updateParametersInContext(context)

# Reinitialize the context to ensure changes are applied
context.reinitialize(preserveState=True)
state = context.getState(getEnergy=True, getPositions=True)
print(context.getParameter("AmoebaVdwLambda"))
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))

simulation.reporters.append(DCDReporter('output.dcd', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=', '))
simulation.step(nSteps)