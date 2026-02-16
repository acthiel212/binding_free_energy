from openmm import *
from openmm.unit import *
import subprocess
from intspan import intspan
import re

path_to_file = os.path.dirname(os.path.abspath(__file__))

def create_Boresch_restraint(restraint_atoms_1, restraint_atoms_2, restraint_constant, restraint_lower_distance, restraint_upper_distance):

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

def create_COM_restraint(restraint_atoms_1, restraint_atoms_2, restraint_constant, restraint_lower_distance, restraint_upper_distance):
    restraint_atoms_1 = list(intspan(restraint_atoms_1))
    restraint_atoms_2 = list(intspan(restraint_atoms_2))
    # OpenMM atom index starts at zero while FFX starts at 1.
    restraint_atoms_1 = [i - 1 for i in restraint_atoms_1]
    restraint_atoms_2 = [i - 1 for i in restraint_atoms_2]

    convert = openmm.KJPerKcal / (openmm.NmPerAngstrom * openmm.NmPerAngstrom)
    restraintEnergy = "lambda_restraints*(step(distance(g1,g2)-u)*k*(distance(g1,g2)-u)^2+step(l-distance(g1,g2))*k*(distance(g1,g2)-l)^2)"
    restraint = openmm.CustomCentroidBondForce(2, restraintEnergy)
    restraint.setForceGroup(0)
    restraint.addGlobalParameter('lambda_restraints', 1.0)
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("l")
    restraint.addPerBondParameter("u")
    restraint.addGroup(restraint_atoms_1)
    restraint.addGroup(restraint_atoms_2)
    restraint.addBond([0, 1], [restraint_constant * convert, restraint_lower_distance * openmm.NmPerAngstrom,
                               restraint_upper_distance * openmm.NmPerAngstrom])

    return restraint

def calculate_restraint_subsection(host_guest_file_path, cutoff, boresch=False, host_name=None, guest_name=None, 
                                   H1=None, H2=None, H3=None, min_adii=None, max_adis=None, l1_range=None, ffx_path=""):
    # Use custom FFX path if provided, otherwise use default
    if ffx_path:
        ffx_executable = ffx_path
    else:
        ffx_executable = f"{path_to_file}/../ffx-1.0.0/bin/ffxc"
    
    with open('RestrainGuest.log', "w") as outfile:
        if boresch:
            # Build Boresch command
            command = [ffx_executable, "test.FindRestraints", "--boresch"]
            if host_name:
                command.extend(["--hostName", host_name])
            if guest_name:
                command.extend(["--guestName", guest_name])
            if H1 is not None:
                command.extend(["--H1", str(H1)])
            if H2 is not None:
                command.extend(["--H2", str(H2)])
            if H3 is not None:
                command.extend(["--H3", str(H3)])
            if min_adii is not None:
                command.extend(["--minAdii", str(min_adii)])
            if max_adis is not None:
                command.extend(["--maxAdis", str(max_adis)])
            if l1_range is not None:
                command.extend(["--l1Range", str(l1_range)])
            command.append(host_guest_file_path)
        else:
            # Original distance cutoff command
            command = [ffx_executable, "test.FindRestraints", "--distanceCutoff", str(cutoff), 
                       host_guest_file_path]
        
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT, text=True)

    # Parse the log output
    if boresch:
        # Parse Boresch guest anchor indices from log
        # Host anchors are provided by the user, so use them directly
        h1_idx = H1
        h2_idx = H2
        h3_idx = H3
        g1_idx = None
        g2_idx = None
        g3_idx = None
        
        with open('RestrainGuest.log', 'r') as infile:
            content = infile.read()
        
        print(f"Debug: FFX log output:\n{content}\n")  # Debug output
        
        # Extract guest anchors only (host anchors are provided by user)
        g1_match = re.search(r'G1:.*?index\s+(\d+)', content, re.DOTALL)
        if not g1_match:
            g1_match = re.search(r'G1[^0-9]*\(.*?index\s+(\d+)', content, re.DOTALL)
        if not g1_match:
            g1_match = re.search(r'G1[^)]*index\s+(\d+)', content)
            
        g2_match = re.search(r'G2:.*?index\s+(\d+)', content, re.DOTALL)
        if not g2_match:
            g2_match = re.search(r'G2[^0-9]*\(.*?index\s+(\d+)', content, re.DOTALL)
        if not g2_match:
            g2_match = re.search(r'G2[^)]*index\s+(\d+)', content)
            
        g3_match = re.search(r'G3:.*?index\s+(\d+)', content, re.DOTALL)
        if not g3_match:
            g3_match = re.search(r'G3[^0-9]*\(.*?index\s+(\d+)', content, re.DOTALL)
        if not g3_match:
            g3_match = re.search(r'G3[^)]*index\s+(\d+)', content)
        
        if g1_match:
            g1_idx = int(g1_match.group(1))
        if g2_match:
            g2_idx = int(g2_match.group(1))
        if g3_match:
            g3_idx = int(g3_match.group(1))
        
        print(f"Debug: User-provided host anchors - H1={h1_idx}, H2={h2_idx}, H3={h3_idx}")
        print(f"Debug: Parsed guest anchors - G1={g1_idx}, G2={g2_idx}, G3={g3_idx}")  # Debug output
        
        # Format as "h1,h2,h3,g1,g2,g3"
        if all(idx is not None for idx in [h1_idx, h2_idx, h3_idx, g1_idx, g2_idx, g3_idx]):
            indices = f"{h1_idx},{h2_idx},{h3_idx},{g1_idx},{g2_idx},{g3_idx}"
            print(f"Boresch anchors: Host(H1={h1_idx}, H2={h2_idx}, H3={h3_idx}), Guest(G1={g1_idx}, G2={g2_idx}, G3={g3_idx})")
        else:
            raise ValueError(f"Failed to parse Boresch guest anchor indices from FFX output. Found: G1={g1_idx}, G2={g2_idx}, G3={g3_idx}")
    else:
        # Parse COM restraint indices
        with open('RestrainGuest.log', 'r') as infile:
            for line in infile:
                if re.search('Restrain list indices:', line):
                    #Grab just the list indices
                    indexString = line.split(':')[1]
                    # Remove brackets and whitespace
                    indices = indexString.replace('[', '').replace(']', '').replace(' ','').strip('\n')
        print(f"Atom indices of guest within specified cutoff of {cutoff} Angstroms that will be restrained: {indices}")

    return indices
