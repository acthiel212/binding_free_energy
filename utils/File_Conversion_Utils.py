import os
import re
import shutil
import subprocess

path_to_file = os.path.dirname(os.path.abspath(__file__))

def append_files_method(file1_path, file2_path):
    with open(file1_path, 'r') as file1:
        with open(file2_path, 'a') as file2:
            shutil.copyfileobj(file1, file2)

def FF_to_XML(file):
    with open('FFtoXML.log', "w") as outfile:
        # Run bundled FFX command to get XML files
        subprocess.run([f"{path_to_file}/../ffx-1.0.0/bin/ffxc", "FFtoXML", f"{file}"],
                       stdout=outfile, text=True)

def save_as_PDB(file):
    print(f"Saving {file} as PDB")
    with open('SaveAsPDB.log', "w") as outfile:
        subprocess.run([f"{path_to_file}/../ffx-1.0.0/bin/ffxc", "Biotype", "-w", "-c", "--name", "LIG", f"{file}"],
                        stdout=outfile, text=True)
    filename=os.path.splitext(file)[0]
    transfer_CONECT_records('SaveAsPDB.log', f"{filename}.pdb")

def save_as_XYZ(file):
    print(f"Saving {file} as XYZ")
    with open('SaveAsXYZ.log', "w") as outfile:
        subprocess.run([f"{path_to_file}/../ffx-1.0.0/bin/ffxc", "SaveAsXYZ", f"{file}"],
                        stdout=outfile, text=True)

def transfer_CONECT_records(file1_path, file2_path):
    with open(file1_path, 'r') as infile, open('CONECT.txt', 'w') as outfile:
        for line in infile:
            if re.search('CONECT', line):
                outfile.write(line)
    append_files_method('CONECT.txt', file2_path)
    os.remove('CONECT.txt')

def merge_host_guest_files(guest_file_path, host_file_path):
    # Count number of HETATM records in Host file
    host_hetatm_number = 0
    guest_name_path = os.path.splitext(guest_file_path)[0]
    host_name_path = os.path.splitext(host_file_path)[0]
    guest_name = os.path.basename(guest_file_path).split('.')[0]
    host_name = os.path.basename(host_file_path).split('.')[0]
    if os.path.exists(f"{guest_name_path}.prm") and os.path.exists(f"{host_name_path}.prm"):
        print(f"Corresponding parameter files: {guest_name_path}.prm and {host_name_path}.prm")
    else:
        print("Parameter files missing for the host or guess. "
              "Please make sure parameter files are located in the same location as these structures!")
        exit()

    # Prepare host file to be merged at top by stripping out CONECT records and counting number of HETATMs
    with open(host_file_path, 'r') as infile, open('host_file_temp.txt', 'w') as outfile:
        for line in infile:
            if re.search('HETATM', line):
                host_hetatm_number += 1
            if re.search('CONECT|END', line):
                line = str("")
            outfile.write(line)

    # Prepare guest file to be merged to host by stripping out CONECT records and offsetting HETATM number by number of host atoms
    with open(guest_file_path, 'r') as infile, open('guest_file_temp.txt', 'w') as outfile:
        for line in infile:
            if re.search('HETATM|TER', line):
                number = line[9:12:1]
                new_atom_num = int(number) + host_hetatm_number
                newline = line[0:8:1] + str(new_atom_num) + line[11:25] + str(2) + line[26:]
                line = newline
            if re.search('CONECT|REMARK', line):
                line = str("")
            outfile.write(line)

    append_files_method('guest_file_temp.txt', 'host_file_temp.txt')
    with open(f"{host_name}_{guest_name}.key", 'w') as keyfile:
        keyfile.write(f"patch {guest_name}.prm\n")
        keyfile.write(f"patch {host_name}.prm\n")

    os.rename("host_file_temp.txt", f"{host_name}_{guest_name}.pdb")
    save_as_XYZ(f"{host_name}_{guest_name}.pdb")
    save_as_PDB(f"{host_name}_{guest_name}.xyz")

    # Remove temporary files
    os.remove('guest_file_temp.txt')
    os.remove('SaveAsPDB.log')
    os.remove('SaveAsXYZ.log')


def copy_XYZ_coords(host_guest_pdb_file, host_guest_xyz_file):
    # Prepare guest file to be merged to host by stripping out CONECT records and offsetting HETATM number by number of host atoms
    xyzfile = open(host_guest_xyz_file)
    xyz_content = xyzfile.readlines()

    with open(host_guest_pdb_file, 'r') as infile, open('host_guest_temp.txt', 'w') as outfile:
        i = 0
        for line in infile:
            if re.search('HETATM', line):
                xyz_line = xyz_content[i+2].strip()
                # Split xyz_line and get (first,second,third) position for (x,y,z) coordinate, cast to float, and truncate to 3 decimal places for pdb format
                x = '%.3f'%(float(str.split(xyz_line)[1]))
                y = '%.3f'%(float(str.split(xyz_line)[2]))
                z = '%.3f'%(float(str.split(xyz_line)[3]))
                line = line[0:31:1] + '%7s'%(x) + line[38:]
                line = line[0:39:1] + '%7s'%(y) + line[46:]
                line = line[0:47:1] + '%7s'%(z) + line[54:]
                i = i + 1
            outfile.write(line)
        os.rename("host_guest_temp.txt", host_guest_pdb_file)