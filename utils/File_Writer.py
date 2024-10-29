import re, os
import shutil

def append_files_method(file1_path, file2_path):
    with open(file1_path, 'r') as file1:
        with open(file2_path, 'a') as file2:
            shutil.copyfileobj(file1, file2)

def transfer_CONECT_records(file1_path, file2_path):
    with open(file1_path, 'r') as infile, open('CONECT.txt', 'w') as outfile:
        for line in infile:
            if re.search('CONECT', line):
                outfile.write(line)
    append_files_method('CONECT.txt', file2_path)
    os.remove('CONECT.txt')