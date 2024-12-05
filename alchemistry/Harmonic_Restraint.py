from openmm import *
import subprocess
from intspan import intspan
import re

path_to_file = os.path.dirname(os.path.abspath(__file__))

def create_restraint(restraint_atoms_1, restraint_atoms_2, restraint_constant, restraint_lower_distance, restraint_upper_distance):
    restraint_atoms_1 = list(intspan(restraint_atoms_1))
    restraint_atoms_2 = list(intspan(restraint_atoms_2))
    # OpenMM atom index starts at zero while FFX starts at 1.
    restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
    restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]

    convert = openmm.KJPerKcal / (openmm.NmPerAngstrom * openmm.NmPerAngstrom)
    restraintEnergy = "step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2"
    restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
    restraint.setForceGroup(0)
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("l")
    restraint.addPerBondParameter("u")
    restraint.addGroup(restraint_atoms_1)
    restraint.addGroup(restraint_atoms_2)
    restraint.addBond([0, 1], [restraint_constant * convert, restraint_lower_distance * openmm.NmPerAngstrom,
                               restraint_upper_distance * openmm.NmPerAngstrom])

    return restraint

def calculate_restraint_subsection(host_guest_file_path, cutoff):
    with open('RestrainGuest.log', "w") as outfile:
        subprocess.run([f"{path_to_file}/../ffx-1.0.0/bin/ffxc", "test.RestrainGuest", "--distanceCutoff",f"{cutoff}",
                        f"{host_guest_file_path}"], stdout=outfile, text=True)

    with open('RestrainGuest.log', 'r') as infile:
        for line in infile:
            if re.search('Restrain list indices:', line):
                #Grab just the list indices
                indexString = line.split(':')[1]
                # Remove brackets and whitespace
                indices = indexString.replace('[', '').replace(']', '').replace(' ','')
    print(f"Atom indices of guest within specified cutoff of {cutoff} Angstroms that will be restrained:", indices)

    return indices
