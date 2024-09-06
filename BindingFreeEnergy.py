from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

pdb = PDBFile('g1.pdb_2')
forcefield = ForceField('g1.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=10*nanometer, constraints=None)
vdwLambda = 0.5
alchemicalAtoms = range(0, 2)

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
context.setParameter("AmoebaVdwLambda", vdwLambda)
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))

#simulation = Simulation(pdb.topology, system, integrator, platform)
#simulation.context.setPositions(pdb.positions)

## Annihilate == 2
## Decouple == 1
vdwForce.setAlchemicalMethod(2)
for atomi in alchemicalAtoms:
    [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(atomi)
    vdwForce.setParticleParameters(atomi, parent, sigma, eps, redFactor, True, type)

vdwForce.updateParametersInContext(context)
context.reinitialize(preserveState=True)
print(context.getParameter("AmoebaVdwLambda"))
state = context.getState(getEnergy=True, getPositions=True)
print(state.getPotentialEnergy().in_units_of(kilocalories_per_mole))


#simulation.reporters.append(PDBReporter('output.pdb', 1000))
#simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=', '))
#simulation.step(3000)


