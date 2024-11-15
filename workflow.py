import os
import shutil
import argparse
import sys

# ----------------------------
# Configuration Parameters
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINDING_FREE_ENERGY_DIR = os.path.join("/Dedicated/schnieders/andthiel/binding_free_energy")

# Template directories
TEMPLATE_GUEST_DIR = os.path.join(SCRIPT_DIR, "Template", "Guest")
TEMPLATE_HOST_GUEST_DIR = os.path.join(SCRIPT_DIR, "Template", "Host_Guest")

# Working directories
WORKING_GUEST_DIR = os.path.join(os.getcwd(), "Guest_Workflow")
WORKING_HOST_GUEST_DIR = os.path.join(os.getcwd(), "Host_Guest_Workflow")

VDW_LAMBDAS = [0, 0.4, 0.5, 0.525, 0.55, 0.575, 0.5875, 0.6, 0.6125, 0.625, 0.6375, 0.65, 0.6625, 
               0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ELEC_LAMBDAS = [0.0] * 22 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# Structure and forcefield files
STRUCTURE_GUEST = os.path.join(TEMPLATE_GUEST_DIR, "g3-00-0000-0000.pdb")
FORCEFIELD_GUEST = os.path.join(TEMPLATE_GUEST_DIR, "hostsG3.xml")
STRUCTURE_HOST_GUEST = os.path.join(TEMPLATE_HOST_GUEST_DIR, "temoa_g3-15-0000-0000.pdb")
FORCEFIELD_HOST_GUEST = os.path.join(TEMPLATE_HOST_GUEST_DIR, "hostsG3.xml")

# Analysis directories
ANALYSIS_GUEST_DIR = os.path.join(WORKING_GUEST_DIR, "analysis")
ANALYSIS_HOST_GUEST_DIR = os.path.join(WORKING_HOST_GUEST_DIR, "analysis")

# Create analysis directories
os.makedirs(ANALYSIS_GUEST_DIR, exist_ok=True)
os.makedirs(ANALYSIS_HOST_GUEST_DIR, exist_ok=True)

# ----------------------------
# User-Defined Flags
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Binding Free Energy Workflow Script")
    parser.add_argument("--start_at", type=str, help="Start the script at the specified method.")
    parser.add_argument("--run_equilibration", type=str, default="true", choices=["true", "false"], help="Whether to run equilibration.")
    parser.add_argument("--alchemical_atoms", type=str, required=True, help="Comma-separated list of alchemical atoms.")
    parser.add_argument("--restraint_atoms1", type=str, help="Comma-separated list of restraint atoms 1.")
    parser.add_argument("--restraint_atoms2", type=str, help="Comma-separated list of restraint atoms 2.")
    
    args = parser.parse_args()
    
    # Validate run_equilibration flag
    if args.run_equilibration not in ["true", "false"]:
        print("Invalid value for --run_equilibration. Use 'true' or 'false'.")
        sys.exit(1)

    # Check if alchemical atoms are provided
    if not args.alchemical_atoms:
        print("Error: You must specify alchemical atoms with --alchemical_atoms.")
        sys.exit(1)

    # Check if restraint atoms are provided
    if not args.restraint_atoms1 or not args.restraint_atoms2:
        print("Error: You must specify both --restraint_atoms1 and --restraint_atoms2.")
        sys.exit(1)

    return args

args = parse_args()

# ----------------------------
# Functions
#----------------------------

class DirectorySetup:
    def __init__(self, binding_free_energy_dir, template_guest_dir, template_host_guest_dir):
        self.binding_free_energy_dir = binding_free_energy_dir
        self.template_guest_dir = template_guest_dir
        self.template_host_guest_dir = template_host_guest_dir

    def setup_directories(self, target_dir, structure_file, forcefield_file, alchemical_atoms,
                          restraint_atoms_1, restraint_atoms_2, workflow_type):
        print(f"Setting up directory: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy job and Python files based on workflow type
        template_dir = self.template_guest_dir if workflow_type == "Guest" else self.template_host_guest_dir
        self.copy_files(template_dir, target_dir, ['thermo.job', 'bar.job'])
        self.copy_files('.', target_dir, [structure_file, forcefield_file])
        os.makedirs(os.path.join(target_dir, 'analysis'), exist_ok=True)
        self.copy_files('.', os.path.join(target_dir, 'analysis'), [structure_file, forcefield_file])

        # Modify job files
        self.modify_job_files(target_dir, structure_file, forcefield_file, alchemical_atoms, 
                              restraint_atoms_1, restraint_atoms_2, workflow_type)

    def copy_files(self, src_dir, dest_dir, files):
        for file in files:
            src_path = os.path.join(src_dir, file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_dir)
            else:
                print(f"Warning: {file} not found in {src_dir}")

    def modify_job_files(self, target_dir, structure_file, forcefield_file, alchemical_atoms,
                         restraint_atoms_1, restraint_atoms_2, workflow_type):
        job_files = ['thermo.job', 'bar.job']
        for job_file in job_files:
            file_path = os.path.join(target_dir, job_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                content = content.replace("<pdb_file>", structure_file)
                content = content.replace("<forcefield_file>", forcefield_file)
                content = content.replace("<alchemical_atoms>", alchemical_atoms)
                
                if workflow_type == "Host_Guest":
                    content = content.replace("<restraint_atoms_1>", restraint_atoms_1)
                    content = content.replace("<restraint_atoms_2>", restraint_atoms_2)

                if "thermo.job" in job_file:
                    content = content.replace("<Production.py>", f"{self.binding_free_energy_dir}/Production.py")
                elif "bar.job" in job_file:
                    content = content.replace("<BAR.py>", f"{self.binding_free_energy_dir}/BAR.py")

                with open(file_path, 'w') as file:
                    file.write(content)


class JobSubmission:
    def __init__(self, binding_free_energy_dir):
        self.binding_free_energy_dir = binding_free_energy_dir

    def submit_equil(self, target_dir):
        equil_job_path = os.path.join(target_dir, 'equil.job')
        shutil.copy(os.path.join(target_dir, 'thermo.job'), equil_job_path)

        with open(equil_job_path, 'r') as file:
            content = file.read()
        content = content.replace('--num_steps 15000000', '--num_steps 5000000')
        content = content.replace("<Equil.py>", f"{self.binding_free_energy_dir}/Equil.py")

        with open(equil_job_path, 'w') as file:
            file.write(content)

        print(f"Submitting equilibration job for {target_dir}")
        equil_job_id = subprocess.check_output(['qsub', '-terse', equil_job_path]).decode().strip()
        return equil_job_id


class LambdaJobManager:
    def __init__(self, vdw_lambdas, elec_lambdas):
        self.vdw_lambdas = vdw_lambdas
        self.elec_lambdas = elec_lambdas

    def submit_thermo(self, target_dir, job_prefix, equil_job_id=None):
        thermo_job_ids = []

        for i, (vdw_lambda, elec_lambda) in enumerate(zip(self.vdw_lambdas, self.elec_lambdas)):
            lambda_dir = os.path.join(target_dir, str(i))
            os.makedirs(lambda_dir, exist_ok=True)

            shutil.copy(os.path.join(target_dir, 'thermo.job'), lambda_dir)
            self.modify_lambda_job(lambda_dir, vdw_lambda, elec_lambda, job_prefix, i)

            cmd = ['qsub', '-terse']
            if equil_job_id:
                cmd.extend(['-hold_jid', equil_job_id])
            cmd.append(os.path.join(lambda_dir, 'thermo.job'))
            job_id = subprocess.check_output(cmd).decode().strip()
            thermo_job_ids.append(job_id)

        return thermo_job_ids

    def modify_lambda_job(self, lambda_dir, vdw_lambda, elec_lambda, job_prefix, index):
        job_file = os.path.join(lambda_dir, 'thermo.job')
        with open(job_file, 'r') as file:
            content = file.read()
        content = content.replace("<vdw_lambda_value>", str(vdw_lambda))
        content = content.replace("<elec_lambda_value>", str(elec_lambda))
        content = content.replace("$JOB_NAME", f"{job_prefix}{index}")

        with open(job_file, 'w') as file:
            file.write(content)


class EnergyCollector:
    @staticmethod
    def collect_energy(target_dir):
        free_energy = 0.0
        log_files = [f for f in os.listdir(target_dir) if f.endswith('.log')]

        for log_file in log_files:
            log_path = os.path.join(target_dir, log_file)
            with open(log_path, 'r') as file:
                for line in file:
                    if "Free energy" in line:
                        energy = float(line.split()[-1])
                        free_energy += energy
        print(f"Total Free Energy: {free_energy}")
        return free_energy