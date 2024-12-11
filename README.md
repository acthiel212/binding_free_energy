# Binding Free Energy Workflow

This repository provides a pipeline for calculating the binding free energy of molecular systems using alchemical free energy methods. The workflow combines molecular dynamics (MD) simulations, free energy perturbation (FEP), and Bennett acceptance ratio (BAR) analysis.

This guide will walk you through the installation process, high-level workflow descriptions, and detailed instructions for running the bash-based pipeline.

---

## Installation

### Prerequisite: Miniconda
First, install Miniconda by following the [official installation guide](https://docs.anaconda.com/miniconda/install/).

### Required Packages
After Miniconda is installed, set up the required environment with the following commands:

1. **OpenMM**  
   Install OpenMM for molecular dynamics simulations:  
   conda install -c conda-forge openmm-setup

2. **pymbar**
    Install pymbar for free energy calculations:
    conda install -c conda-forge "pymbar<4"

3. **mdtraj**
    Install MDTraj for trajectory analysis:
    conda install -c conda-forge mdtraj

4. **pdbfixer**
    Install pdbfixer for fixing PDB files:
    conda install -c conda-forge pdbfixer

5. **intspan**
    Install intspan for handling intervals in the workflow:
    pip install intspan

6. **Force Field X**
    Force Field X is bundled in this repository. If a new version is needed, download it from the official Force Field X website.


# Binding Free Energy Workflow Overview

This section provides a detailed breakdown of the workflow for calculating binding free energy using alchemical free energy methods. Follow the steps below to execute the pipeline successfully.

---

## Input Files

You will need two sets of input files for the simulation:

- **Guest molecule**:
  - `guest_mol.pdb`: This file contains the atomic coordinates of the guest molecule.  
  - `guest_mol.xml`: This file contains the force field parameters specific to the guest molecule.

- **Host-guest complex**:
  - `host_guest_mol.pdb`: This file contains the atomic coordinates of the host molecule, guest molecule, water, and ions in the system.  
  - `host_guest_mol.xml`: This file contains pre-generated force field parameters for the host, water, and ions.  







ex: 
output1: python BindingFreeEnergy.py 

python BAR.py --traj_i output1.dcd --traj_ip1 output2.dcd --pdb_file temoa_g3-15-0000-0000.pdb --forcefield_file hostsG3.xml  --vdw_lambda_i 0 --elec_lambda_i 0 --vdw_lambda_ip1 0.4 --elec_lambda_ip1 0 --alchemical_atoms "0,2" --nonbonded_method NoCutoff

