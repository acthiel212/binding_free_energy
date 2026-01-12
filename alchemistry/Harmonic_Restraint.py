from openmm import *
from openmm.unit import *
import subprocess
from intspan import intspan
import re

path_to_file = os.path.dirname(os.path.abspath(__file__))

def create_restraint(restraint_atoms_1, restraint_atoms_2, restraint_constant, restraint_lower_distance, restraint_upper_distance):

    # Set bond restraint properties 
    bond_energy_function = "lambda_restraints*(K/2)*(r-r0)^2;"
    harmonicforce=CustomBondForce(bond_energy_function)
    harmonicforce.addPerBondParameter('r0')
    harmonicforce.addPerBondParameter('K')
    harmonicforce.addGlobalParameter('lambda_restraints', 1.0)
  
    # Set angle restraint properties 
    angle_energy_function = "lambda_restraints*0.5*k*(theta-theta0)^2"
    angleforce=CustomAngleForce(angle_energy_function)
    angleforce.addPerAngleParameter('theta0')
    angleforce.addPerAngleParameter('k')
    angleforce.addGlobalParameter('lambda_restraints', 1.0)
 
    # Set torsion restraint properties 
    torsion_energy_function = "lambda_restraints*k*(1+cos(n*theta-theta0))"
    torsionforce=CustomTorsionForce(torsion_energy_function)
    torsionforce.addPerTorsionParameter('n')
    torsionforce.addPerTorsionParameter('theta0')
    torsionforce.addPerTorsionParameter('k')
    torsionforce.addGlobalParameter('lambda_restraints', 1.0)

    # Add bond
    bondlength=6.2*unit_definitions.angstroms
    bondforce=10.0*unit_definitions.kilocalorie_per_mole/unit_definitions.angstroms**2
    harmonicforce.addBond(10,219, [bondlength,bondforce])

    # Add angles

    restrainedangle=0.76*unit_definitions.radians
    angleconst=500*unit_definitions.kilocalorie_per_mole/unit_definitions.radian**2
    angleforce.addAngle(62,10,219,[restrainedangle,angleconst])

    restrainedangle=2.01*unit_definitions.radians
    angleconst=500*unit_definitions.kilocalorie_per_mole/unit_definitions.radian**2
    angleforce.addAngle(10,219,225,[restrainedangle,angleconst])

    # Add torsion

    restrainedtorsion=(-0.57-3.14)*unit_definitions.radians
    torsionconst=500*unit_definitions.kilocalorie_per_mole
    torsionforce.addTorsion(124,62,10,219,[1, restrainedtorsion,torsionconst])

    restrainedtorsion=(-1.02-3.14)*unit_definitions.radians
    torsionconst=500*unit_definitions.kilocalorie_per_mole
    torsionforce.addTorsion(62,10,219,225,[1, restrainedtorsion,torsionconst])

    restrainedtorsion=(-2.28-3.14)*unit_definitions.radians
    torsionconst=500*unit_definitions.kilocalorie_per_mole
    torsionforce.addTorsion(10,219,225,224,[1, restrainedtorsion,torsionconst])

    return harmonicforce, angleforce, torsionforce

def calculate_restraint_subsection(host_guest_file_path, cutoff):
    with open('RestrainGuest.log', "w") as outfile:
        subprocess.run([f"{path_to_file}/../ffx-1.0.0/bin/ffxc", "test.FindRestraints", "--distanceCutoff",f"{cutoff}",
                        f"{host_guest_file_path}"], stdout=outfile, text=True)

    with open('RestrainGuest.log', 'r') as infile:
        for line in infile:
            if re.search('Restrain list indices:', line):
                #Grab just the list indices
                indexString = line.split(':')[1]
                # Remove brackets and whitespace
                indices = indexString.replace('[', '').replace(']', '').replace(' ','').strip('\n')
    print(f"Atom indices of guest within specified cutoff of {cutoff} Angstroms that will be restrained: {indices}")

    return indices
