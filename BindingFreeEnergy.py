# BindingFreeEnergy.py
import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from intspan import intspan
from sys import stdout

# Argument parser for user-defined flags
def parse_arguments():
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
    return parser.parse_args()

# Create and setup the system for simulation
def setup_system(pdb_file, forcefield_file, nonbonded_method, nonbonded_cutoff):
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(forcefield_file)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded_method, nonbondedCutoff=nonbonded_cutoff, constraints=None)
    return system, pdb

# Create the restraint force
def create_restraint(system, args, restraint_atoms_1, restraint_atoms_2):
    if args.use_restraints:
        convert = openmm.KJPerKcal / (openmm.NmPerAngstrom * openmm.NmPerAngstrom)
        restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
        restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
        restraint.setForceGroup(0)
        restraint.addPerBondParameter("k")
        restraint.addPerBondParameter("l")
        restraint.addPerBondParameter("u")
        restraint.addGroup(restraint_atoms_1)
        restraint.addGroup(restraint_atoms_2)
        restraint.addBond([0, 1], [args.restraint_constant, args.restraint_lower_distance, args.restraint_upper_distance])
        system.addForce(restraint)

# Set up integrator and simulation
def setup_simulation(pdb, system, step_size):
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond)
    platform = Platform.getPlatformByName('CUDA')
    return Simulation(pdb.topology, system, integrator, platform)


# Set parameters for alchemical atoms
def set_alchemical_parameters(vdwForce, multipoleForce, alchemical_atoms, vdw_lambda, elec_lambda, context):
    vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)
    vdwForce.updateParametersInContext(context)

    for i in alchemical_atoms:
        params = multipoleForce.getMultipoleParameters(i)
        charge = params[0] * elec_lambda
        dipole = [d * elec_lambda for d in params[1]]
        quadrupole = [q * elec_lambda for q in params[2]]
        polarizability = params[-1] * elec_lambda
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
    multipoleForce.updateParametersInContext(context)

# Run the simulation and log energy
def run_simulation(simulation, nSteps, pdb, vdw_lambda, output_file):
    simulation.context.setPositions(pdb.getPositions())
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.context.setParameter("AmoebaVdwLambda", vdw_lambda)
    
    with open(output_file, 'w') as f:
        simulation.reporters.append(DCDReporter('output3.dcd', 1000))
        simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=', ', elapsedTime=True))
        
        for step in range(1, nSteps + 1):
            simulation.step(1)
            if step % 1000 == 0:
                state = simulation.context.getState(getEnergy=True)
                potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                f.write(f"{potential_energy}\n")
    print("Simulation complete and energies saved to", output_file)

def main():
    args = parse_arguments()
    
    # Parse alchemical_atoms input
    alchemical_atoms = list(intspan(args.alchemical_atoms))
    
    # Set nonbonded method
    nonbonded_method_map = {'NoCutoff': NoCutoff, 'CutoffNonPeriodic': CutoffNonPeriodic, 'PME': PME, 'Ewald': Ewald}
    nonbonded_method = nonbonded_method_map.get(args.nonbonded_method, NoCutoff)
    
    # Load system and pdb file
    system, pdb = setup_system(args.pdb_file, args.forcefield_file, nonbonded_method, args.nonbonded_cutoff * nanometer)
    
    # Create restraints if necessary
    if args.use_restraints:
        restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
        restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
        create_restraint(system, args, restraint_atoms_1, restraint_atoms_2)
    
    # Set up simulation
    simulation = setup_simulation(pdb, system, args.step_size)

    context = simulation.context
    
    # Identify forces
    numForces = system.getNumForces()
    vdwForce, multipoleForce = None, None
    for i in range(numForces):
        force = system.getForce(i)
        if isinstance(force, AmoebaVdwForce):
            vdwForce = force
        elif isinstance(force, AmoebaMultipoleForce):
            multipoleForce = force
    
    if vdwForce is None or multipoleForce is None:
        raise ValueError("AmoebaVdwForce or AmoebaMultipoleForce not found.")
    
    # Set alchemical parameters
    set_alchemical_parameters(vdwForce, multipoleForce, alchemical_atoms, args.vdw_lambda, args.elec_lambda, context)
    
    # Run simulation and save energies
    run_simulation(simulation, args.num_steps, pdb, args.vdw_lambda, 'binding_energies.txt')

if __name__ == '__main__':
    main()
