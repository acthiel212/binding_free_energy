import argparse

def create_default_parser():
    parser = argparse.ArgumentParser(description='OpenMM General Setup with Custom Flags')
    parser.add_argument('--pdb_file', required=True, type=str, help='PDB file for the simulation')
    parser.add_argument('--forcefield_file', required=True, type=str, help='Force field XML file')
    parser.add_argument('--nonbonded_method', required=True, type=str,
                        help='Nonbonded method: NoCutoff, CutoffNonPeriodic, PME, etc.')
    parser.add_argument('--nonbonded_cutoff', required=False, type=float,
                        help='Nonbonded cutoff in nm (default: 1.0 nm)', default=1.0)
    return parser

def add_dynamics_parser(parser):
    parser.add_argument('--num_steps', required=False, type=int, help='Number of MD steps to take.', default=1000)
    parser.add_argument('--step_size', required=False, type=int, help='Step size given to integrator in fs.', default=1)

    parser.add_argument('--name_dcd', type=str, default='output.dcd',
                        help='Specify the output DCD filename (default: output.dcd)')

    parser.add_argument('--checkpoint_freq', type=int, default=1000,
                        help='Frequency (in steps) to save checkpoints (default: 1000)')
    parser.add_argument('--checkpoint_prefix', type=str, default='checkpoint',
                        help='Prefix for checkpoint filenames (default: checkpoint)')
    parser.add_argument('--output_pdb', type=str, help='PDB file to write to at completion', default='output.pdb')
    parser.add_argument('--output_xml', type=str, help='XML file of system state to write to at completion', default='output.xml')
    return parser

def add_restraint_parser(parser):
    parser.add_argument('--use_restraints', required=False, type=bool, help='Whether to use restraint', default=False)
    parser.add_argument('--restraint_atoms_1', required=False, type=str,
                        help='Range of atoms in restraint group 1 (e.g., "0,2")', default="")
    parser.add_argument('--restraint_atoms_2', required=False, type=str,
                        help='Range of atoms in restraint group 2 (e.g., "0,2")', default="")
    parser.add_argument('--restraint_constant', required=False, type=float,
                        help='Restraint force constant (default: 1.0)', default=1.0)
    parser.add_argument('--restraint_lower_distance', required=False, type=float,
                        help='Restraint lower distance (default: 0.0)', default=0.0)
    parser.add_argument('--restraint_upper_distance', required=False, type=float,
                        help='Restraint upper distance (default: 1.0)', default=1.0)
    return parser

def add_alchemical_parser(parser):
    # Argument parser for user-defined flags
    parser.add_argument('--vdw_lambda', required=False, type=float,
                        help='Value for van der Waals lambda (default: 1.0)', default=1.0)
    parser.add_argument('--elec_lambda', required=False, type=float,
                        help='Value for electrostatic lambda (default: 0.0)', default=0.0)
    parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
    return parser

def add_BAR_parser(parser):
    parser.add_argument('--traj_i', required=True, type=str, help='DCD file for lambda i')
    parser.add_argument('--traj_ip1', required=True, type=str, help='DCD file for lambda i+1')
    parser.add_argument('--alchemical_atoms', required=True, type=str, help='Range of alchemical atoms (e.g., "0,2")')
    parser.add_argument('--vdw_lambda_i', required=True, type=float, help='Lambda for van der Waals at state i')
    parser.add_argument('--elec_lambda_i', required=True, type=float, help='Lambda for electrostatics at state i')
    parser.add_argument('--vdw_lambda_ip1', required=True, type=float, help='Lambda for van der Waals at state i+1')
    parser.add_argument('--elec_lambda_ip1', required=True, type=float, help='Lambda for electrostatics at state i+1')
    # New flags for traversing the DCD file
    parser.add_argument('--step_size', type=int, required=False, help='Step size to traverse the DCD file', default=1)
    parser.add_argument('--start', type=int, required=False, help='Start frame for DCD traversal', default=0)
    parser.add_argument('--stop', type=int, required=False, help='Stop frame for DCD traversal', default=None)
    return parser