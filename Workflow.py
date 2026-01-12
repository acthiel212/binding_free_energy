import argparse
import os
import sys
import subprocess
import re
import shutil
from pathlib import Path
from shutil import copyfile

# ----------------------------
# Configuration Parameters
# ----------------------------
from alchemistry.Harmonic_Restraint import calculate_restraint_subsection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINDING_FREE_ENERGY_DIR = SCRIPT_DIR # Assuming workflow.py exists in binding_free_energy

VDW_LAMBDAS = [
    0, 0.4, 0.5, 0.525, 0.55, 0.575, 0.5875, 0.6, 0.6125, 0.625, 0.6375, 0.65, 0.6625, 0.675, 0.7,
    0.725, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
]
ELEC_LAMBDAS = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
]
RESTRAINT_LAMBDAS = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0000, 0.4217, 0.1778, 0.0750, 0.0316, 0.0133, 0.0056, 0.0024, 0.0010, 0.0000
]

# ----------------------------
# User-Defined Flags
# ----------------------------


# Default values for user-defined flags
CWD = os.getcwd()
TEMPLATE_GUEST_DIR = ""
TEMPLATE_HOST_GUEST_DIR = ""
WORKING_GUEST_DIR = ""
WORKING_HOST_GUEST_DIR = ""
ANALYSIS_GUEST_DIR = ""
ANALYSIS_HOST_GUEST_DIR = ""
NAME = ""
ALCHEMICAL_ATOMS = ""
RESTRAINT_ATOMS_1 = ""
RESTRAINT_ATOMS_2 = ""
START_AT = ""
RUN_EQ = True
SETUP_ONLY = False
SUB_TYPE = ""

def parse_arguments():
    """Parse command-line arguments and initialize global workflow directories."""
    global NAME, ALCHEMICAL_ATOMS, RESTRAINT_ATOMS_1, RESTRAINT_ATOMS_2, START_AT, RUN_EQ, SETUP_ONLY, SUB_TYPE,\
        TEMPLATE_GUEST_DIR, TEMPLATE_HOST_GUEST_DIR, WORKING_GUEST_DIR, WORKING_HOST_GUEST_DIR, \
        ANALYSIS_GUEST_DIR, ANALYSIS_HOST_GUEST_DIR

    parser = argparse.ArgumentParser(description="Script options for the workflow.")

    # Define command-line arguments
    parser.add_argument("--guest_name", type=str, default="", required=True, help="Name of guest molecule. (Should match the name of the XYZ file.)")
    parser.add_argument("--start_at", type=str, default="", help="Start the script at the specified method. Available methods: setup_directories, submit_equil, submit_thermo, submit_bar, collect_energy.")
    parser.add_argument("--setup_only", type=str, choices=["true", "false"], default="false",
                        help="Whether to only setup directories and files without submitting jobs. Defaults to false.")
    parser.add_argument("--run_equilibration", type=str, choices=["true", "false"], default="true",
                        help="Whether to run equilibration. Defaults to true.")
    parser.add_argument("--alchemical_atoms", type=str, default="", help="Comma-separated list of alchemical atoms (required). "
                                                                            "If not set, automatically determines set from guest atoms.")
    parser.add_argument("--restraint_atoms1", type=str, default="", help="Comma-separated list of restraint atoms 1 (optional). If not set, automatically choose restraints.")
    parser.add_argument("--restraint_atoms2", type=str, default="", help="Comma-separated list of restraint atoms 2 (optional). If not set, automatically choose restraints.")
    parser.add_argument("--submission_system",type=str,default="SGE",help="Submission system to submit jobs to. Modifying this only affects the search for job files that that selected submission system would expect. Only SGE and SLURM are currently supported.")
    # Parse the arguments
    args = parser.parse_args()

    # Assign values from the parsed arguments
    NAME = args.guest_name
    START_AT = args.start_at
    SETUP_ONLY = args.setup_only.lower() == "true"
    RUN_EQ = args.run_equilibration.lower() == "true"
    SUB_TYPE = args.submission_system.upper()
    if args.alchemical_atoms == "":
        guest_dir = os.path.join(BINDING_FREE_ENERGY_DIR, "Guests")
        with open(f"{guest_dir}/{NAME}.xyz", 'r') as f:
            first_line = f.readline()
            first_column = first_line.split()[0]

        args.alchemical_atoms = f"1-{first_column}"

    ALCHEMICAL_ATOMS = args.alchemical_atoms
    RESTRAINT_ATOMS_1 = args.restraint_atoms1
    RESTRAINT_ATOMS_2 = args.restraint_atoms2
    # Template directories
    TEMPLATE_GUEST_DIR = os.path.join(SCRIPT_DIR, "workflow", f"{NAME}", "Template", "Guest")
    TEMPLATE_HOST_GUEST_DIR = os.path.join(SCRIPT_DIR, "workflow", f"{NAME}", "Template", "Host_Guest")

    # Working directories
    WORKING_GUEST_DIR = os.path.join(SCRIPT_DIR, "workflow", f"{NAME}", "Guest_Workflow")
    WORKING_HOST_GUEST_DIR = os.path.join(SCRIPT_DIR, "workflow", f"{NAME}", "Host_Guest_Workflow")

    # Analysis directories
    ANALYSIS_GUEST_DIR = os.path.join(WORKING_GUEST_DIR, "analysis")
    ANALYSIS_HOST_GUEST_DIR = os.path.join(WORKING_HOST_GUEST_DIR, "analysis")

    os.makedirs(ANALYSIS_GUEST_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_HOST_GUEST_DIR, exist_ok=True)

    return args


# ----------------------------
# Functions
#----------------------------

def replace_in_file(file_path, replacements):
    """Replace placeholders in a file with specified values."""
    with open(file_path, "r") as f:
        content = f.read()
    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)
    with open(file_path, "w") as f:
        f.write(content)

def run_prepare():
    """Execute Prepare.py to set up guest and host-guest structure files."""
    # ----------------------------
    # Prepare Step
    # ----------------------------
    print("Running Prepare.py...")
    os.chdir(os.path.join(BINDING_FREE_ENERGY_DIR, "Guests"))
    prepare_command = [
        "python", os.path.join(BINDING_FREE_ENERGY_DIR, "Prepare.py"),
        "--guest_file", f"{NAME}.xyz",
        "--prm_file", f"{NAME}.prm",
        "--host_file_dir", os.path.join(BINDING_FREE_ENERGY_DIR, "HP-BCD"),
        "--target_dir", os.path.join(BINDING_FREE_ENERGY_DIR, "workflow"),
        "--docked_file", os.path.join(BINDING_FREE_ENERGY_DIR, "DockedStructures", f"HPBCD_1_{NAME}.results.xyz"),
        "--job_file_dir", os.path.join(BINDING_FREE_ENERGY_DIR, f"JobFiles/{SUB_TYPE}")
    ]

    subprocess.run(prepare_command, check=True)
    print("Prepare step completed.")

def setup_directories(target_dir, structure_file, forcefield_file, alchemical_atoms,
                      restraint_atoms_1, restraint_atoms_2, workflow_type):
    """
    Sets up the directory and job files for the specified workflow type.

    Args:
        target_dir (str): Path to the target directory to set up.
        structure_file (str): Path to the structure (PDB) file.
        forcefield_file (str): Path to the force field file.
        alchemical_atoms (str): Alchemical atoms string.
        restraint_atoms_1 (str): Restraint atoms set 1.
        restraint_atoms_2 (str): Restraint atoms set 2.
        workflow_type (str): Workflow type, either 'Guest' or 'Host_Guest'.
    """

    # Determine the template directory based on workflow type
    if workflow_type == "Guest":
        template_dir = TEMPLATE_GUEST_DIR
    elif workflow_type == "Host_Guest":
        template_dir = TEMPLATE_HOST_GUEST_DIR
        if restraint_atoms_1 == "":
            restraint_atoms_1 = "1-217"
        if restraint_atoms_2 == "":
            current_directory = os.getcwd()
            os.chdir(TEMPLATE_HOST_GUEST_DIR)
            restraint_atoms_2 = calculate_restraint_subsection(f"{TEMPLATE_HOST_GUEST_DIR}/{structure_file}", 5.0)
            os.chdir(current_directory)
    else:
        raise ValueError("Invalid workflow type. Must be 'Guest' or 'Host_Guest'.")

    if not template_dir:
        raise ValueError(f"Template directory for workflow type '{workflow_type}' is not defined.")

    # Ensure the template directory exists
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")

    # ----------------------------
    # Solvate Step
    # ----------------------------
    print("Running Solvate.py...")
    os.chdir(template_dir)
    solvate_command = [
        "python", os.path.join(BINDING_FREE_ENERGY_DIR, "Solvate.py"),
        "--pdb_file", structure_file,
        "--forcefield_file", f"{NAME}.xml", "hp-bcd.xml",
        "--nonbonded_method", "PME"
    ]
    subprocess.run(solvate_command, check=True)
    os.chdir(CWD)

    print("Solvate step completed.")

    print(f"Setting up production directories: {target_dir}")

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    def safe_copy(src, dest):
        """Copy file from src to dest if src exists."""
        if os.path.exists(src):
            print(f"Source exists: {src}")  # Debugging line
            shutil.copy(src, dest)
            print(f"Copied {src} to {dest}")
        else:
            print(f"File not found: {src}. Skipping copy.")


    # List of files to copy
    if template_dir == TEMPLATE_HOST_GUEST_DIR:
        solv_min_struct = f"hp-bcd_{NAME}_solv_min.pdb"
    else:
        solv_min_struct = f"{NAME}_solv_min.pdb"
    file_list = [
        "thermo.job",
        "bar.job",
        "equil.job",
        solv_min_struct,
    ]

    # Create the analysis subdirectory
    analysis_dir = os.path.join(target_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Copy files from the template directory to the target and analysis directories
    for file in file_list:
        src_path = os.path.join(template_dir, file)
        print(f"Checking file: {src_path}")
        safe_copy(src_path, target_dir)
        safe_copy(src_path, analysis_dir)

    # Modify job files in the target directory
    for job_file in ["thermo.job", "bar.job", "equil.job"]:
        job_path = os.path.join(target_dir, job_file)
        if not os.path.exists(job_path):
            continue
        replacements = {
            "<pdb_file>": f"{target_dir}/{solv_min_struct}",
            "<forcefield_file>": f"{template_dir}/{forcefield_file}",
            "<alchemical_atoms>": alchemical_atoms,
            "Name": NAME,
            "hp-bcd.xml": f"{template_dir}/hp-bcd.xml"   #parameter file for host
            }

        if workflow_type == "Host_Guest":
            replacements.update({
                "<restraint_atoms_1>": restraint_atoms_1,
                "<restraint_atoms_2>": restraint_atoms_2,
            })

        replace_in_file(job_path, replacements)
    
    print(f"Alchemical atoms: {alchemical_atoms}")
    print(f"Restraint atoms 1: {restraint_atoms_1}")
    print(f"Restraint atoms 2: {restraint_atoms_2}")
    print(f"Force field file: {forcefield_file}")
	
    print(f"Directory and files set up for {target_dir}")

def submit_equil(target_dir, job_prefix):
    """
    Submits the equilibration job after updating the equil.job file with the correct path to Equil.py.

    Args:
        target_dir (str): Path to the target directory where the equilibration job will run.
        job_prefix (str): Prefix used for job submission.

    Returns:
        str: Job ID of the submitted equilibration job.
    """
    print(f"Inside submit_equil function\nTarget directory: {target_dir}\nJob prefix: {job_prefix}")

    equil_job_path = os.path.join(target_dir, "equil.job")
    if not os.path.exists(equil_job_path):
        print("Equilibration job file not found.")
        return None

    # Ensure BINDING_FREE_ENERGY_DIR is set and resolve Equil.py's path
    binding_free_energy_dir = BINDING_FREE_ENERGY_DIR
    if not binding_free_energy_dir:
        raise EnvironmentError("BINDING_FREE_ENERGY_DIR environment variable is not set.")

    equil_py_path = os.path.join(binding_free_energy_dir, "Equil.py")
    if not os.path.exists(equil_py_path):
        raise FileNotFoundError(f"Equil.py not found in {binding_free_energy_dir}.")

    # Replace placeholder in equil.job with the correct path to Equil.py
    replace_in_file(equil_job_path, {
        "<Equil.py>": equil_py_path
    })

    # Submit the job
    print("Submitting equilibration job...")
    os.chdir(target_dir)
    if SUB_TYPE == "SGE" and not SETUP_ONLY :
        try:
            equil_job_id = subprocess.check_output(["qsub", "-terse", "equil.job"]).decode().strip()
            print(f"Submitted equilibration job {equil_job_id} for {target_dir}")
        finally:
            os.chdir("..")  # Ensure the working directory is reset
        return equil_job_id


thermo_job_ids_guest = []
thermo_job_ids_host_guest = []


def submit_thermo(target_dir, job_prefix, equil_job_id):
    """
    Function to create lambda directories and submit thermo jobs.
    :param target_dir: Directory where lambda subdirectories will be created.
    :param job_prefix: Prefix for the job name.
    :param equil_job_id: Job ID to hold on if specified.
    """
    global thermo_job_ids_guest, thermo_job_ids_host_guest

    # Assuming VDW_LAMBDAS and ELEC_LAMBDAS are defined globally
    num_lambdas = len(VDW_LAMBDAS)

    vdw_lambdas = VDW_LAMBDAS
    print(f"Setting up lambda directories{target_dir}", file=sys.stderr)
    if job_prefix == "OMM_Guest_LAM":
        vdw_lambdas = VDW_LAMBDAS[:-10]
    for i, vdw_lambda in enumerate(vdw_lambdas):
        lambda_dir = os.path.join(target_dir, str(i))
        os.makedirs(lambda_dir, exist_ok=True)

        # Copy and modify thermo.job
        thermo_job_path = os.path.join(target_dir, "thermo.job")
        target_job_path = os.path.join(lambda_dir, "thermo.job")
        with open(thermo_job_path, "r") as f:
            job_content = f.read()
        
        job_content = job_content.replace("<vdw_lambda_value>", str(vdw_lambda))
        job_content = job_content.replace("<elec_lambda_value>", str(ELEC_LAMBDAS[i]))
        if job_prefix != "OMM_Guest_LAM":
            job_content = job_content.replace("<restraint_lambda_value>", str(RESTRAINT_LAMBDAS[i]))
        job_content = job_content.replace("<Production.py>", str(os.path.join(BINDING_FREE_ENERGY_DIR, "Production.py")))
        job_content = job_content.replace("_LAM", f"_LAM{i}")

        with open(target_job_path, "w") as f:
            f.write(job_content)

        # Submit job
        if SUB_TYPE == "SGE" and not SETUP_ONLY:
            # Change directory to lambda_dir for job submission
            os.chdir(lambda_dir)

            print(
                f"Submitting thermo job for lambda directory {lambda_dir} "
                f"with vdw_lambda={vdw_lambda}, elec_lambda={ELEC_LAMBDAS[i]}",
                file=sys.stderr
            )

            if equil_job_id:
                command = ["qsub", "-terse", "-hold_jid", equil_job_id, target_job_path]
            else:
                command = ["qsub", "-terse", target_job_path]

            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

            # Check for submission errors
            if result.returncode != 0 or not result.stdout.strip():
                print(f"Failed to submit thermo job for {lambda_dir}", file=sys.stderr)
                exit(1)

            thermo_job_id = result.stdout.strip()

            # Navigate back to the target directory
            os.chdir(target_dir)

            # Store thermo job ID in the appropriate list
            if job_prefix == "OMM_Guest_LAM":
                thermo_job_ids_guest.append(thermo_job_id)
            else:
                thermo_job_ids_host_guest.append(thermo_job_id)

            print(thermo_job_id, file=sys.stderr)


def safe_symlink(target, link_name):
    """Create a symlink, replacing any existing file or symlink with the same name."""
    if os.path.exists(link_name) or os.path.islink(link_name):
        os.unlink(link_name)  # Remove existing file or symlink
    os.symlink(target, link_name)

def submit_bar(target_dir, analysis_type, thermo_job_ids):

    """
    Create and submit BAR jobs for the given lambda pairs.

    Args:
        target_dir (str): The target directory containing lambda directories and bar.job file.
        analysis_type (str): Type of analysis ("host-guest" or other).
        thermo_job_ids (list): List of job IDs from the thermo phase.

    Returns:
        list: List of submitted BAR job IDs.
    """
    num_lambdas = len(VDW_LAMBDAS)
    bar_job_ids = []

    os.makedirs(f"{target_dir}/analysis", exist_ok=True)
    print(f"Setting up BAR jobs for {target_dir} with thermo job IDs: {thermo_job_ids} and analysis type: {analysis_type}")
    print(f"Number of lambdas: {num_lambdas}")

    for i in range(num_lambdas - 1):
        lambda_dir_i = os.path.join(target_dir, str(i))
        lambda_dir_next = os.path.join(target_dir, str(i + 1))

        # Validate existence of lambda directories
        if not os.path.isdir(lambda_dir_i) or not os.path.isdir(lambda_dir_next):
            print(f"Warning: Lambda directories {lambda_dir_i} or {lambda_dir_next} do not exist. Skipping BAR job for pair {i} and {i + 1}.")
            continue

        # Move and link output DCD files based on the analysis type
        if analysis_type == "host-guest":
            safe_symlink(os.path.realpath(f"{lambda_dir_i}/output.dcd"), f"{ANALYSIS_HOST_GUEST_DIR}/output_{i}.dcd")
            safe_symlink(os.path.realpath(f"{lambda_dir_next}/output.dcd"), f"{ANALYSIS_HOST_GUEST_DIR}/output_{i + 1}.dcd")
            bar_job_file = f"{ANALYSIS_HOST_GUEST_DIR}/bar.job_{i}"
            shutil.copyfile(f"{target_dir}/bar.job", bar_job_file)
        else:
            safe_symlink(os.path.realpath(f"{lambda_dir_i}/output.dcd"), f"{ANALYSIS_GUEST_DIR}/output_{i}.dcd")
            safe_symlink(os.path.realpath(f"{lambda_dir_next}/output.dcd"), f"{ANALYSIS_GUEST_DIR}/output_{i + 1}.dcd")
            bar_job_file = f"{ANALYSIS_GUEST_DIR}/bar.job_{i}"
            shutil.copyfile(f"{target_dir}/bar.job", bar_job_file)

        # Set vdw and elec lambda values
        vdw_lambda = VDW_LAMBDAS[i]
        vdw_lambda_ip1 = VDW_LAMBDAS[i + 1]
        elec_lambda = ELEC_LAMBDAS[i]
        elec_lambda_ip1 = ELEC_LAMBDAS[i + 1]
        restraint_lambda = RESTRAINT_LAMBDAS[i]
        restraint_lambda_ip1 = RESTRAINT_LAMBDAS[i + 1]

        # Replace placeholders in the bar.job file
        with open(bar_job_file, 'r') as file:
            job_content = file.read()

        job_content = job_content.replace('<traj_i>', f"{target_dir}/analysis/output_{i}.dcd")
        job_content = job_content.replace('<traj_ip1>', f"{target_dir}/analysis/output_{i + 1}.dcd")
        job_content = job_content.replace('<vdw_lambda_value_i>', str(vdw_lambda))
        job_content = job_content.replace('<vdw_lambda_value_ip1>', str(vdw_lambda_ip1))
        job_content = job_content.replace('<elec_lambda_value_i>', str(elec_lambda))
        job_content = job_content.replace('<elec_lambda_value_ip1>', str(elec_lambda_ip1))
        if analysis_type != "guest":
            job_content = job_content.replace('<restraint_lambda_value_i>', str(restraint_lambda))
            job_content = job_content.replace('<restraint_lambda_value_ip1>', str(restraint_lambda_ip1))
        job_content = job_content.replace("<BAR.py>", str(os.path.join(BINDING_FREE_ENERGY_DIR, "BAR.py")))
        job_content = job_content.replace('$LAMBDA_I', str(i))
        job_content = job_content.replace('$LAMBDA_NEXT', str(i + 1))

        with open(bar_job_file, 'w') as file:
            file.write(job_content)

        # Submit job based on START_AT and analysis type
        if SUB_TYPE == "SGE" and not SETUP_ONLY:
            try:
                os.chdir(f"{target_dir}/analysis")
                if START_AT == "submit_bar" or not thermo_job_ids:
                    result = subprocess.run(
                        ['qsub', '-terse', bar_job_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        universal_newlines=True
                    )
                    bar_job_id = result.stdout.strip()
                    print(
                        f"Submitted BAR job {bar_job_id} for lambda pair {i} and {i + 1} in {analysis_type} analysis without dependencies")
                else:
                    # Check that thermo_job_ids has sufficient entries for i and i + 1
                    if i < len(thermo_job_ids) - 1:
                        result = subprocess.run(
                            ['qsub', '-terse', '-hold_jid', f"{thermo_job_ids[i]},{thermo_job_ids[i + 1]}",
                             bar_job_file],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                            universal_newlines=True
                        )
                        bar_job_id = result.stdout.strip()
                        print(
                            f"Submitted BAR job {bar_job_id} for lambda pair {i} and {i + 1} in {analysis_type} analysis with dependencies")
                    else:
                        print(f"Skipping BAR job for lambda pair {i} and {i + 1} due to insufficient thermo_job_ids.")
                        continue

                bar_job_ids.append(bar_job_id)
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit BAR job for lambda pair {i} and {i + 1}: {e.stderr}")
                continue
            finally:
                os.chdir('..')
    return bar_job_ids

def collect_energy(target_dir):
    """Collect and sum free energy values from log files in the target directory."""
    target_dir = Path(target_dir)
    log_files = target_dir.glob("*.log")
    free_energy = 0.0

    print(f"Collecting free energy values from {target_dir}")
    for log_file in log_files:
        with log_file.open() as f:
            content = f.read()
            # Regex pattern to extract energy value (modify as per your log format)
            match = re.search(r"Free energy: ([\d\.\-]+)", content)
            if match:
                energy_value = float(match.group(1))
                free_energy += energy_value
                print(f"Collected free energy {energy_value} from {log_file.name}")
            else:
                print(f"No free energy value found in {log_file.name}")

    print(f"Total free energy collected from {target_dir}: {free_energy}")
    return free_energy

# Define the workflow execution function
def execute_workflow(start_method):
    """Execute the binding free energy workflow starting from the specified method."""
    methods_order = [
        "setup_directories",
        "submit_equil",
        "submit_thermo",
        "submit_bar",
        "collect_energy"
    ]

    start_found = False
    equil_job_ids = {}
    thermo_job_ids_guest = []
    thermo_job_ids_host_guest = []

    if "setup_directories" in methods_order and not start_found:
        print("Running the prepare step before setting up directories...")
        run_prepare()

    for method in methods_order:
        if not start_found:
            if method == start_method:
                start_found = True
            else:
                continue

        print(f"Executing method: {method}")



        if method == "setup_directories":
            setup_directories(WORKING_GUEST_DIR, f"{NAME}.pdb", f"{NAME}.xml", ALCHEMICAL_ATOMS, "", "", "Guest")

            # Adjust alchemical atom indices for host-guest
            start, end = map(int, ALCHEMICAL_ATOMS.split("-"))
            alch_start_host_guest = start + 217
            alch_end_host_guest = end + 217
            alchemical_atoms_host_guest = f"{alch_start_host_guest}-{alch_end_host_guest}"

            setup_directories(
                WORKING_HOST_GUEST_DIR,
                f"hp-bcd_{NAME}.pdb",
                f"{NAME}.xml",
                alchemical_atoms_host_guest,
                RESTRAINT_ATOMS_1,
                RESTRAINT_ATOMS_2,
                "Host_Guest")

        elif method == "submit_equil":
            equil_job_ids = {}
            # Submit jobs for guest and host-guest
            equil_job_ids["guest"] = submit_equil(WORKING_GUEST_DIR, "OMM_Guest_LAM")
            equil_job_ids["host_guest"] = submit_equil(WORKING_HOST_GUEST_DIR, "OMM_Host_Guest_LAM")
            print(equil_job_ids)

        elif method == "submit_thermo":
            guest_dep = equil_job_ids.get("guest", "")
            host_guest_dep = equil_job_ids.get("host_guest", "")

            thermo_job_ids_guest = submit_thermo(WORKING_GUEST_DIR, "OMM_Guest_LAM", guest_dep)
            thermo_job_ids_host_guest = submit_thermo(WORKING_HOST_GUEST_DIR, "OMM_Host_Guest_LAM", host_guest_dep)

        elif method == "submit_bar":
            if start_method == "submit_bar":
                # Skip dependency check for thermo jobs
                guest_bar_jobs = submit_bar(WORKING_GUEST_DIR, "guest", [])
                host_guest_bar_jobs = submit_bar(WORKING_HOST_GUEST_DIR, "host-guest", [])
            else:
                guest_bar_jobs = submit_bar(WORKING_GUEST_DIR, "guest", thermo_job_ids_guest)
                host_guest_bar_jobs = submit_bar(WORKING_HOST_GUEST_DIR, "host-guest", thermo_job_ids_host_guest)

            # Print the submitted BAR job IDs for both analyses
            print(f"Guest BAR jobs: {guest_bar_jobs}")
            print(f"Host-Guest BAR jobs: {host_guest_bar_jobs}")


        elif method == "collect_energy":
            guest_energy = collect_energy(f"{WORKING_GUEST_DIR}/analysis")
            host_guest_energy = collect_energy(f"{WORKING_HOST_GUEST_DIR}/analysis")

            # Calculate binding energy
            binding_energy = host_guest_energy - guest_energy
            print(f"Final binding energy: {binding_energy}")
            print("All jobs submitted successfully.")

        else:
            raise ValueError(f"Unknown method: {method}")

# Main execution
# Ensure parse_arguments() is called to initialize flags
parse_arguments()
if START_AT:
    if START_AT not in ["setup_directories", "submit_equil", "submit_thermo", "submit_bar", "collect_energy"]:
        raise ValueError(f"Invalid method for --start_at. Available methods: setup_directories, submit_equil, submit_thermo, submit_bar, collect_energy")
    print(f"Starting execution at method: {START_AT}")
    execute_workflow(START_AT)
else:
    print("Starting full workflow.")
    execute_workflow("setup_directories")
