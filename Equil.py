#!/usr/bin/env python

import argparse
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from intspan import intspan

def main():
    # Argument parser for user-defined flags
    parser = argparse.ArgumentParser(description='OpenMM Equilibration and Energy Minimization Script')
    parser.add_argument('--pdb_file', required=True, type=str, help='PDB file for the simulation')
    parser.add_argument('--forcefield_file', required=True, type=str, help='Force field XML file')
    parser.add_argument('--nonbonded_method', required=True, type=str, help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
    parser.add_argument('--num_steps', required=False, type=int, help='Number of MD steps for equilibration (default: 10000)', default=10000)
    parser.add_argument('--step_size', required=False, type=int, help='Step size given to integrator in fs (default: 2)', default=2)
    parser.add_argument('--nonbonded_cutoff', required=False, type=float, help='Nonbonded cutoff in nm (default: 1.0 nm)', default=1.0)
    parser.add_argument('--vdw_lambda', required=False, type=float, help='Value for van der Waals lambda (default: 1.0)', default=1.0)
    parser.add_argument('--elec_lambda', required=False, type=float, help='Value for electrostatic lambda (default: 0.0)', default=0.0)
    parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
    parser.add_argument('--use_restraints', required=False, type=bool, help='Whether to use restraint (default: False)', default=False)
    parser.add_argument('--restraint_atoms_1', required=False, type=str, help='Range of atoms in restraint group 1 (e.g., "0,2")', default="")
    parser.add_argument('--restraint_atoms_2', required=False, type=str, help='Range of atoms in restraint group 2 (e.g., "0,2")', default="")
    parser.add_argument('--restraint_constant', required=False, type=float, help='Restraint force constant (default: 1.0)', default=1.0)
    parser.add_argument('--restraint_lower_distance', required=False, type=float, help='Restraint lower distance (default: 0.0)', default=0.0)
    parser.add_argument('--restraint_upper_distance', required=False, type=float, help='Restraint upper distance (default: 1.0)', default=1.0)
    parser.add_argument('--checkpoint_frequency', required=False, type=int, help='Frequency of checkpoint saving in steps (default: 1000)', default=1000)
    parser.add_argument('--checkpoint_file', required=False, type=str, help='Name of the checkpoint file', default='checkpoint.chk')
    parser.add_argument('--minimized_pdb', type=str, default='minimized.pdb', help='Output PDB file for minimized structure (default: minimized.pdb)')

    args = parser.parse_args()

    # Parse alchemical_atoms input
    alchemical_atoms = list(intspan(args.alchemical_atoms))
    alchemical_atoms = [i - 1 for i in alchemical_atoms]  # Adjust for OpenMM indexing

    # Restraint setup if enabled
    use_restraint = args.use_restraints
    if use_restraint:
        if not args.restraint_atoms_1 or not args.restraint_atoms_2:
            raise ValueError("Restraint atoms must be specified if restraints are enabled.")
        restraint_atoms_1 = list(intspan(args.restraint_atoms_1))
        restraint_atoms_2 = list(intspan(args.restraint_atoms_2))
        restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
        restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]
        restraint_constant = args.restraint_constant
        restraint_lower_distance = args.restraint_lower_distance
        restraint_upper_distance = args.restraint_upper_distance

    # Simulation parameters
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

    # Create restraint force if applicable
    if use_restraint:
        restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2 + step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
        restraint = CustomCentroidBondForce(2, restraintEnergy)
        restraint.setForceGroup(0)
        restraint.addPerBondParameter("k")
        restraint.addPerBondParameter("l")
        restraint.addPerBondParameter("u")
        restraint.addGroup(restraint_atoms_1)
        restraint.addGroup(restraint_atoms_2)
        restraint.addBond([0, 1], [restraint_constant, restraint_lower_distance, restraint_upper_distance])
        system.addForce(restraint)
        print("Adding Restraint with parameters: ", restraint.getBondParameters(0))
        restraint.setUsesPeriodicBoundaryConditions(True)
        print("Using PBC Conditions on Restraint? ", restraint.usesPeriodicBoundaryConditions())

    # Identify required forces
    numForces = system.getNumForces()
    forceDict = {}
    for i in range(numForces):
        force_name = system.getForce(i).__class__.__name__
        forceDict[force_name] = i
    print("Available forces in the system:", forceDict)

    # Ensure the required forces exist
    required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce']
    for force_name in required_forces:
        if force_name not in forceDict:
            raise ValueError(f"Required force '{force_name}' not found in the system.")

    vdwForce = system.getForce(forceDict['AmoebaVdwForce'])
    vdwForce.setForceGroup(1)
    multipoleForce = system.getForce(forceDict['AmoebaMultipoleForce'])
    multipoleForce.setForceGroup(1)

    # Setup simulation context
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, step_size*femtosecond)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'double'}
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)

    checkpoint_file = args.checkpoint_file

    # Check if checkpoint file exists
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        simulation.loadCheckpoint(checkpoint_file)
    else:
        print("Initializing simulation from PDB file...")
        simulation.context.setPositions(pdb.getPositions())
        simulation.context.setVelocitiesToTemperature(300*kelvin)
        simulation.context.setParameter("AmoebaVdwLambda", vdw_lambda)

    # Alchemical method setup for van der Waals force
    vdwForce = system.getForce(forceDict['AmoebaVdwForce'])
    vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
    for i in alchemical_atoms:
        params = vdwForce.getParticleParameters(i)
        # Assuming the parameters are (parent, sigma, epsilon, redFactor, isAlchemical, type)
        parent, sigma, epsilon, redFactor, isAlchemical, type = params
        vdwForce.setParticleParameters(i, parent, sigma, epsilon, redFactor, True, type)
    vdwForce.updateParametersInContext(simulation.context)

    # Apply electrostatic scaling to multipole force
    multipoleForce = system.getForce(forceDict['AmoebaMultipoleForce'])
    for i in alchemical_atoms:
        params = multipoleForce.getMultipoleParameters(i)
        charge = params[0] * elec_lambda
        dipole = [d * elec_lambda for d in params[1]]
        quadrupole = [q * elec_lambda for q in params[2]]
        polarizability = params[-1] * elec_lambda
        # Update multipole parameters (keeping other parameters unchanged)
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
    multipoleForce.updateParametersInContext(simulation.context)

    # Add CheckpointReporter
    checkpoint_frequency = args.checkpoint_frequency
    simulation.reporters.append(CheckpointReporter(checkpoint_file, checkpoint_frequency))

    print(f"Starting equilibration for {nSteps} steps...")
    
    # Perform equilibration
    simulation.step(nSteps)
    
    print("Equilibration completed.")

    # Perform energy minimization
    print("Starting energy minimization...")
    tolerance = 10 * kilojoule / nanometer / mole
    simulation.minimizeEnergy(tolerance=tolerance, maxIterations=0, reporter=None)
    print("Energy minimization completed.")

    # Save minimized structure
    minimized_pdb = args.minimized_pdb
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(minimized_pdb, 'w'))
    print(f"Minimized structure saved to {minimized_pdb}.")

if __name__ == "__main__":
    main()
