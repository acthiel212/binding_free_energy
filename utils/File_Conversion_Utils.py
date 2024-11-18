import re, os
import subprocess
import shutil

def append_files_method(file1_path, file2_path):
    with open(file1_path, 'r') as file1:
        with open(file2_path, 'a') as file2:
            shutil.copyfileobj(file1, file2)

def FFtoXML(file):
    with open('FFtoXML.log', "w") as outfile:
        subprocess.run(["ffx-1.0.0/bin/ffxc", "FFtoXML", "-w", "-c", "--name", "LIG", f"{file}"],
                       stdout=outfile, text=True)

def SaveAsPDB(file):
    print(f"Saving {file} as PDB")
    with open('SaveAsPDB.log', "w") as outfile:
        subprocess.run(["ffx-1.0.0/bin/ffxc", "Biotype", "-w", "-c", "--name", "LIG", f"{file}"],
                        stdout=outfile, text=True)
    filename=os.path.splitext(file)
    transfer_CONECT_records('SaveAsPDB.log', f"{filename}.pdb")

def transfer_CONECT_records(file1_path, file2_path):
    with open(file1_path, 'r') as infile, open('CONECT.txt', 'w') as outfile:
        for line in infile:
            if re.search('CONECT', line):
                outfile.write(line)
    append_files_method('CONECT.txt', file2_path)
    os.remove('CONECT.txt')

def merge_host_guest_files(file1_path, file2_path):
    return