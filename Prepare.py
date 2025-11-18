import argparse
import os
import shutil

from utils.File_Conversion_Utils import save_as_PDB, FF_to_XML, merge_host_guest_files, copy_XYZ_coords

parser = argparse.ArgumentParser(description='Preparation and setup of files for running in OpenMM')
parser.add_argument('--guest_file', required=True, type=str, help='Guest XYZ file output by PolType')
parser.add_argument('--prm_file', required=True, type=str, help='PRM file for the guest output by PolType')
parser.add_argument('--host_file_dir', required=True, type=str, help='Directory that contains the PDB and XML already prepared for the host')
parser.add_argument('--target_dir', required=True, type=str, help='Directory to put prepared files into')
parser.add_argument('--docked_file', required=True, type=str, help='Directory to grab docked XYZ coordinates from')
parser.add_argument('--job_file_dir', required=True, type=str, help='Directory to pull job files from')

args = parser.parse_args()
target_dir = args.target_dir
HP_dir = args.host_file_dir
job_dir = args.job_file_dir
CWD = os.getcwd()

# Extract filename without extension
filename = os.path.splitext(args.guest_file)[0]
print(f"Preparing {filename}.")

# Run the ffxc command and redirect output to temp.log
save_as_PDB(args.guest_file)

# Extract specific lines from temp.log and write to corresponding log files
with open("SaveAsPDB.log", "r") as temp_log:
    with open("biotype.log", "w") as biotype_log:
        for line in temp_log:
            if "biotype" in line:
                biotype_log.write(line)

# Create directories if they do not exist
guest_dir = f"{target_dir}/{filename}/Template/Guest"
os.makedirs(guest_dir, exist_ok=True)

# Copy XML file from HP_dir
shutil.copy(f"{job_dir}/guest_bar.job", f"{guest_dir}/bar.job")
shutil.copy(f"{job_dir}/guest_equil.job", f"{guest_dir}/equil.job")
shutil.copy(f"{job_dir}/guest_thermo.job", f"{guest_dir}/thermo.job")
shutil.copy(f"{HP_dir}/hp-bcd.xml", guest_dir)
shutil.copy(f"{HP_dir}/hp-bcd.pdb", guest_dir)

# Combine files and save in the target directory
with open(f"{guest_dir}/{filename}.prm", "w") as out_prm:
    with open(args.prm_file, "r") as prm_file:
        out_prm.write(prm_file.read())
    with open("biotype.log", "r") as biotype_log:
        out_prm.write(biotype_log.read())

# Run the ffxc FFtoXML command
FF_to_XML(f"{guest_dir}/{filename}.prm")

# Remove temporary log files
os.remove("SaveAsPDB.log")
os.remove("biotype.log")
shutil.move(f"{filename}.pdb", f"{guest_dir}/{filename}.pdb")

# Create another directory and copy necessary files for Host_Guest
host_guest_dir = f"{target_dir}/{filename}/Template/Host_Guest"
os.makedirs(host_guest_dir, exist_ok=True)

# Copy over needed files from guest dir
shutil.copy(f"{job_dir}/host_guest_bar.job", f"{host_guest_dir}/bar.job")
shutil.copy(f"{job_dir}/host_guest_equil.job", f"{host_guest_dir}/equil.job")
shutil.copy(f"{job_dir}/host_guest_thermo.job", f"{host_guest_dir}/thermo.job")
shutil.copy(f"{guest_dir}/{filename}.pdb", f"{host_guest_dir}/{filename}.pdb")
shutil.copy(f"{guest_dir}/{filename}.prm", f"{host_guest_dir}/{filename}.prm")
shutil.copy(f"{HP_dir}/hp-bcd.pdb", f"{host_guest_dir}/hp-bcd.pdb")
shutil.copy(f"{HP_dir}/hp-bcd.prm", host_guest_dir)
shutil.copy(f"{HP_dir}/hp-bcd.xml", host_guest_dir)
shutil.copy(f"{guest_dir}/{filename}.xml", host_guest_dir)

# Combine guest and hp-bcd files
os.chdir(host_guest_dir)
merge_host_guest_files(f"{host_guest_dir}/{filename}.pdb", f"{host_guest_dir}/hp-bcd.pdb")
copy_XYZ_coords(f"{host_guest_dir}/hp-bcd_{filename}.pdb", f"{args.docked_file}")
#os.remove(f"{filename}.prm"), os.remove("hp-bcd.prm")
os.chdir(CWD)

