from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

def run_alchemical_simulation(pdb, system, integrator, platform, nSteps, vdw_lambda, elec_lambda, alchemical_atoms, output_prefix='output'):
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # The simulation context manages positions, velocities, and forces
    context = simulation.context
    context.setPositions(pdb.getPositions())
    context.setVelocitiesToTemperature(300 * kelvin)

    # Controls how much the van der Waals interactions are scaled during the alchemical transformation
    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    
    # Force dictionary
    force_dict = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force_name = force.__class__.__name__
        force_dict[force_name] = i

    # Update van der Waals parameters for alchemical atoms
    vdwForce = system.getForce(force_dict['AmoebaVdwForce'])
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
    vdwForce.updateParametersInContext(context)

    # Apply alchemical scaling to the multipole force
    multipoleForce = system.getForce(force_dict['AmoebaMultipoleForce'])
    # The charge, dipole, quadrupole, and polarizability are scaled by elec_lambda to reduce/eliminate electrostatic interactions during the alchemical transformation
    for i in alchemical_atoms:
        params = multipoleForce.getMultipoleParameters(i)
        charge = params[0] * elec_lambda
        dipole = [d * elec_lambda for d in params[1]]
        quadrupole = [q * elec_lambda for q in params[2]]
        polarizability = params[-1] * elec_lambda
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
    multipoleForce.updateParametersInContext(context)

    # Reinitialize the context
    context.reinitialize(preserveState=True)

    # Setup reporters
    simulation.reporters.append(DCDReporter(f'{output_prefix}.dcd', 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True, separator=', '))

    # Run the simulation
    simulation.step(nSteps)

    # Save energies for BAR analysis
    state = context.getState(getEnergy=True, getPositions=True)
    with open(f'energies_{vdw_lambda}_{elec_lambda}.txt', 'w') as f:
        f.write(f'{state.getPotentialEnergy().in_units_of(kilocalories_per_mole)}\n')
