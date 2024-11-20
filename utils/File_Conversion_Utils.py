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
    guest_name = os.path.splitext(guest_file_path)[0]
    host_name = os.path.splitext(host_file_path)[0]
    if os.path.exists(f"{guest_name}.prm") and os.path.exists(f"{host_name}.prm"):
        print(f"Corresponding parameter files: {guest_name}.prm and {host_name}.prm")
    else:
        print("Parameter files missing for the host or guess. "
              "Please make sure parameter files are located in the same location as these structures!")
        exit()


    with open(host_file_path, 'r') as infile, open('host_file_temp.txt', 'w') as outfile:
        for line in infile:
            if re.search('HETATM', line):
                host_hetatm_number += 1
            if re.search('CONECT|END', line):
                line = str("")
            outfile.write(line)

    # Read in the file
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
    #os.remove('SaveAsPDB.log')
    #os.remove('SaveAsXYZ.log')



merge_host_guest_files("Diflunisal.pdb", "hp-bcd.pdb")